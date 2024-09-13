import json
import numpy as np
import os
import queue
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from models.transkit import build
from flask import Flask, request
from utils.logs import setup_logger
from addict import Dict as adict
from models.experimental import attempt_load
from utils.yolo.general import check_img_size
import torch
import processor
from utils.transkit.misc import load_conf
from utils.yolo.torch_utils import select_device
from utils.yolo.torch_utils import TracedModel

logger = setup_logger("receiver")

app = Flask(__name__)

app.config.from_file("config.json", load=json.load)

# Ensure the upload and processed folders exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

if not os.path.exists(app.config["PROCESSED_FOLDER"]):
    os.makedirs(app.config["PROCESSED_FOLDER"])

yolo, transkit = None, None

"""
Queue and ThreadPool for processing

The idea is to have a receiver endpoint that saves the received chunks to disk and sends their paths to a message queue.
Then, one or more threads receive these chunk paths, pull and analyze those chunks, and write their metadata next to the chunk.
You can adjust the number of workers based on your requirements and needs.
Note: Disk I/O is slower, but keeping data in memory could lead to OOM (Out Of Memory) errors if the analysis is slow.
"""

message_queue = queue.Queue()
num_worker_threads = 4  # UPF - Adjust based on requirements
executor = ThreadPoolExecutor(max_workers=num_worker_threads)


def start_processor():
    while True:
        message = message_queue.get()
        if message is None:
            break
        executor.submit(processor.process_chunk, message)


# Start the processor thread
processor_thread = threading.Thread(target=start_processor)
processor_thread.start()


@app.route('/health')
def health():
    return "OK", 200


@app.route('/videos/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    segmentdir = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
    if os.path.exists(segmentdir):
        # Delete the directory containing the video and metadata
        shutil.rmtree(segmentdir)
        return {"message": f"Video {video_id} deleted successfully"}, 200
    else:
        return {"error": f"Video {video_id} not found"}, 404


@app.route('/<id>/status', methods=['GET'])
def status(id):
    segmentdir = os.path.join(app.config['UPLOAD_FOLDER'], id)
    if not os.path.exists(segmentdir):
        return {"error": "ID not found"}, 404

    metadata_file = os.path.join(segmentdir, app.config["METADATA_FILE"])

    logger.debug(f"searching for {metadata_file}")

    if os.path.exists(metadata_file):
        return {"status": "processed"}, 200
    else:
        return {"status": "processing"}, 202


# Retrieve metadata file from ID directory
@app.route('/<id>/metadata', methods=['GET'])
def get_metadata(id):
    segmentdir = os.path.join(app.config['UPLOAD_FOLDER'], id)
    metadata_file_path = os.path.join(segmentdir, app.config['METADATA_FILE'])

    logger.debug(f"searching for {metadata_file_path}")

    if not os.path.exists(metadata_file_path):
        return {"error": "Metadata not found"}, 404

    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)

    return metadata, 200


# Receive video, generate ID, save chunk with ID, and return ID
@app.route('/upload', methods=['POST'])
def upload_video_chunk():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not file:
        return "File upload failed", 500

    global yolo, transkit
    if yolo is None or transkit is None:

        device = select_device(app.config["DEVICE"])
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load YOLO
        yolo_conf = adict(app.config["YOLO"])
        yolo = attempt_load(yolo_conf.weights_file, map_location=device)  # load FP32 model
        stride = int(yolo.stride.max())  # model stride
        checked_imgsz = check_img_size(yolo_conf.image_size, s=stride)  # check img_size
        yolo = TracedModel(yolo, device, yolo_conf.image_size)
        if half:
            yolo.half()  # to FP16
        yolo.conf = yolo_conf

        # Load TransKit
        device = torch.device(app.config["DEVICE"])
        transkit_conf = load_conf(adict(app.config["TRANSKIT"]))
        transkit = build(transkit_conf).to(device)
        checkpoint = torch.load(transkit_conf.weights_file, map_location=device)
        transkit.load_state_dict(checkpoint['model_state_dict'])
        transkit.eval()

    extension = os.path.splitext(file.filename)[1]
    id = str(uuid.uuid4())  # Generate a unique ID
    segmentdir = os.path.join(app.config['UPLOAD_FOLDER'], id)
    os.makedirs(segmentdir)
    dir_filename = os.path.join(segmentdir, id) + extension
    file.save(dir_filename)

    # Add the file path to the queue for processing
    message_queue.put((dir_filename,
                       yolo,
                       transkit,
                       app.config["DEVICE"]))

    return {"id": id}, 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
