# UAVlabels

A semi-automatic UAV target annotation tool developed based on labelImg.

[English](./README.md) | [简体中文](./README.zh.md)

## Overview

UAVlabels is a tool optimized for UAV image annotation, inheriting the core annotation features from labelImg while enhancing and optimizing for drone object detection scenarios. (Currently only the YOLO format is completed)

## System Requirements

- Python 3.10

## Installation Guide

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd UAVlabels
```

### 2. Install Dependencies

The project uses `requirements.txt` to manage dependencies. Run the following command to install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Download Models

Obtain the ONNX file for UAV detection and ORTrack tracking weights from the link below:
`https://pan.baidu.com/s/1FHcaWI0PSyaesn59TnmBXw Extraction code: qcuy`

After downloading, place the model files in the `autolabeling/models` directory.<br>
Edit `autolabeling/config/models.json` to define the ONNX model.

## Usage Guide

### 1. Launch the Application

After installation, run the following command to start the application:

```bash
python labelImg.py
```

The graphical user interface will launch, and you can begin annotating UAV images. (Currently only the YOLO format is completed)

### 2. Auto Annotation

<p align="center">
<img width="822" height="444" alt="image" src="https://github.com/user-attachments/assets/36e9dc68-294c-41b2-9319-1741e32ee240" /><br>
Click the "Auto Annotation" menu option to access the auto-annotation interface, and select either `Detection` or `Tracking` functionality.




### 3. Detection Feature

<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/fb963621-9fd8-4379-9ec8-9cbb755d52a8" /><br>

Select a defined ONNX model from `Detection Models`, choose the detection scope (single frame or all frames in a folder), and modify the annotation file save directory.


### 4. Tracking Feature

<p align="center">
<img width="40%" alt="image" src="https://github.com/user-attachments/assets/a28ed4b4-b8c3-4c9a-9402-d85ac2a85d47" /><br>

Customize the IOU threshold: when the tracking result and the annotation box in the original image have an IOU higher than the threshold, choose to:<br>
Keep the original annotation box or use the tracking result as the annotation box.<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/c687d660-83b9-4b14-aed7-5de5b778fecc" /><br>

Select either `CSRT` or `ORTrack` as the tracker. When using `ORTrack`, you can optionally enable GPU acceleration.<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/d5d7f3db-845d-4fad-8ddb-7b83beec0a89" /><br>

Select any target as the starting target and click 'Start Tracking' to begin auto-annotation.<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/890deb62-7d29-466d-8d4d-59fc56edbc0d" /><br>

Begin visual tracking. If you detect tracking errors, press 'Stop Tracking' to stop.<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/83325a9f-33ed-462d-bf21-77456a9a4cf8" /><br>

Enter the save interface. Use AD keys to navigate between frames. When you are satisfied with a tracking frame, press Enter to save all tracking results up to that frame.<br>



## Key Features

- Integrates semi-automatic annotation methods based on both object detection and object tracking paradigms
- Loads ONNX models for detection
- Integrates CSRT and ORTrack methods for UAV tracking


## License

Developed based on the labelImg project

## Contributing

We welcome issues and pull requests to help improve this project.

## Acknowledgements / Citation

This project utilizes the following open-source projects and algorithms:

*   **CSRT Tracker**: The implementation is based on the OpenCV library [1], which is licensed under the Apache 2.0 License. The underlying algorithm is from the paper:
    > A. Lukezic, et al. "Discriminative Correlation Filter with Channel and Spatial Reliability." *IJCV*, 2018. [4†L11-L12]

*   **ORTrack**: The implementation is based on the official repository [2] and the paper:
    > [Author Names], "Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking." *CVPR*, 2025. [1†L28]

[1] OpenCV: https://opencv.org/
[2] ORTrack GitHub Repository: https://github.com/wuyou3474/ORTrack
---
