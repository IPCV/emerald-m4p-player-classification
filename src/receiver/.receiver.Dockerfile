# Use a Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

RUN pip install gdown

# Copy the requirements file and install dependencies
COPY receiver/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY receiver/receiver.py receiver.py
COPY receiver/processor.py processor.py
COPY receiver/transforms.py transforms.py
COPY receiver/models models

# Copy the utils directory into the container
COPY receiver/utils utils
# RUN mkdir -p utils
# COPY receiver/utils/torch_utils.py utils
COPY utils/logs.py utils/logs.py

# Copy the uploads directory into the container
COPY receiver/uploads uploads

# Copy config.json file into the container
COPY receiver/config.json config.json

RUN mkdir -p processed

WORKDIR /app/receiver/weights
RUN gdown --no-check-certificate https://drive.google.com/uc?id=1TCT0QehMQVDDM5Mx2mgxK3lvumvJzwmQ
RUN gdown --no-check-certificate https://drive.google.com/uc?id=12Z-yiKp304Ko2ah-Ch6vX_Gtd8pGB91b

# Command to run the application
WORKDIR /app
CMD ["python", "receiver.py"]
