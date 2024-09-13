import cv2
import json
import os
import random
import time
import torch
from addict import Dict as adict
from torch.utils.data import DataLoader
from utils.logs import setup_logger

from models.experimental import attempt_load
from models.transkit import build
from transforms import transform_image
from utils.transkit.data import CollateFrames
from utils.transkit.data import make_seq_mask
from utils.transkit.data import unmask_flat_sequences
from utils.transkit.datasets import PatchesDataset
from utils.transkit.misc import load_conf
from utils.yolo.datasets import LoadImages
from utils.yolo.general import check_img_size
from utils.yolo.general import non_max_suppression
from utils.yolo.general import scale_coords
from utils.yolo.plots import plot_one_box
from utils.yolo.torch_utils import TracedModel
from utils.yolo.torch_utils import select_device

logger = setup_logger("Processor")


def save_metadata_as_json(metadata, file_path):
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def extract_frames(video_path, sample_rate=2.):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        logger.error("Error: Could not open video.")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    interval = int(fps / sample_rate)
    frame_count = 0

    frames, frame_ids = [], []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frames.append(frame)
            frame_ids.append(frame_count)

        frame_count += 1

    video_capture.release()
    return frames, frame_ids


def detect(frames, yolo, device):
    device = select_device(device)
    half = device.type != 'cpu'

    stride = int(yolo.stride.max())  # model stride
    checked_imgsz = check_img_size(yolo.conf.image_size, s=stride)  # check img_size
    dataset = LoadImages(frames, img_size=checked_imgsz, stride=stride)

    # Get names and colors
    names = yolo.module.names if hasattr(yolo, 'module') else yolo.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        yolo(torch.zeros(1, 3, yolo.conf.image_size, yolo.conf.image_size).to(device).type_as(
            next(yolo.parameters())))  # run once
    old_img_w = old_img_h = yolo.conf.image_size
    old_img_b = 1

    detections = {i: [] for i in range(len(dataset))}
    for num, (img, im0s) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                yolo(img, augment=yolo.conf.augment)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = yolo(img, augment=yolo.conf.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, yolo.conf.conf_thres, yolo.conf.iou_thres, classes=yolo.conf.classes,
                                   agnostic=yolo.conf.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0, frame = im0s, getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    detections[num].append([int(xy) for xy in xyxy])

                    # label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Save results (image with detections)
            # cv2.imwrite(f'/app/processed/detections_{num}.jpg', im0)
    return detections


def classify_players(frames, detections, model, device):
    loader = DataLoader(dataset=PatchesDataset(frames, detections),
                        collate_fn=CollateFrames(),
                        shuffle=False,
                        batch_size=2,
                        num_workers=1)

    predictions = []
    with torch.no_grad():
        device = torch.device(device)
        for batch_idx, (players, bboxes, lengths) in enumerate(loader):
            players = players.to(device)
            seq_mask = make_seq_mask(lengths).to(device)
            outputs = model(players, None, src_key_padding_mask=seq_mask)
            prediction = outputs.view(-1, model.transformer.num_classes).argmax(dim=1)
            prediction = unmask_flat_sequences(prediction, lengths)

            for id, num_dets in enumerate(lengths):
                predictions.append([{'bb': bboxes[id][i], 'class': int(prediction[id][i])} for i in range(num_dets)])
    return predictions


def process_chunk(args):

    video_path, yolo, transkit, device = args
    time0 = time.time()
    frames, frame_ids = extract_frames(video_path)
    for i, frame in enumerate(frames):
        cv2.imwrite(f'/app/processed/{i}.jpg', frame)

    detections = detect(frames, yolo, device)
    predictions = classify_players(frames, detections, transkit, device)
    results = {frame_id: predictions[i] for i, frame_id in enumerate(frame_ids)}
    time1 = time.time()

    logger.debug(f"Metadata exported: {results}")
    metadata = {
        "status": "processed",
        "video_path": video_path,
        "prediction": results
    }
    metadata_file_path = os.path.join(os.path.split(video_path)[0], 'metadata.json')
    logger.debug(f"Saving metadata file to {metadata_file_path}")
    save_metadata_as_json(metadata, metadata_file_path)
    time2 = time.time()

    logger.debug(f'Total time : {time2 - time0} - Process: {time1 - time0} - Metadata: {time2 - time1}')
