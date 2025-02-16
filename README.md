# Custom Object Detection with YOLOv5

## Overview

This repository contains a YOLOv5-based object detection pipeline for identifying cats and dogs, incorporating a custom bounding box similarity metric. The dataset is included in the repository, and all dependencies for training and inference are specified in the Jupyter notebook.

## Prerequisites

- Python 3.8

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SasininduSV/Custom-Object-Detection-YOLO.git
   cd Custom-Object-Detection-YOLO
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## YOLOv5 Setup & Architecture

- The model used is **YOLOv5-small**.
- The architecture follows the standard YOLOv5 structure, optimized for lightweight and efficient object detection.

## Custom Metric Definition

- A new similarity measure for bounding box comparison is implemented in the `metric.py` script.
- This metric enhances the evaluation of object detection performance beyond standard IoU-based measures.

## Training & Evaluation Instructions

1. Follow the \`\` notebook for model training and evaluation.
2. Training is performed using the YOLOv5 framework, and hyperparameters can be modified in the training command:
   ```bash
   python yolov5/train.py --img 640 --batch 16 --epochs 100 --data dataset/data.yaml --weights yolov5s.pt --device 0
   ```
3. Evaluation can be done using:
   ```bash
   python yolov5/val.py --weights yolov5/runs/train/exp/weights/best.pt --data dataset/data.yaml --img 640 --task test --device 0
   ```
4. Key hyperparameters like learning rate, batch size, and data augmentation can be adjusted in the training script or configuration files.

## Notebook Instructions

- The \`\` notebook includes detailed steps for:
  - Dependency installation.
  - Data preprocessing.
  - Model training and hyperparameter tuning.
  - Evaluation using the custom metric.

## Dataset

- The dataset required for training and evaluation is included in the repository.
- Ensure the dataset structure follows the expected YOLOv5 format (images and labels in respective directories).

## Inference

- To run inference on new images, use:
  ```bash
  python detect.py --weights runs/train/custom_yolov5/weights/best.pt --img 640 --source path_to_image_or_folder
  python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --source /path/to/image.jpg --img 640 --conf 0.4
  ```
