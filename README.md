# UAVlabels

基于labelImg开发的无人机目标半自动化标注工具。

## 功能介绍

UAVlabels是一款专为无人机图像标注优化的工具，继承了labelImg的核心标注功能，并针对无人机目标检测场景进行了增强和优化。（现在只完成了yolo格式部分）

## 系统要求

- Python 3.10

## 安装指南

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd UAVlabels
```

### 2. 安装依赖

项目使用`requirements.txt`管理依赖，执行以下命令安装所有所需的包：

```bash
pip install -r requirements.txt
```
### 3. 下载模型

从链接获得用于无人机检测的onnx文件与ORTrack跟踪权重
`https://pan.baidu.com/s/1FHcaWI0PSyaesn59TnmBXw 提取码: qcuy `
下载后将模型文件放入`autolabeling/models` 中<br>
编辑`autolabeling/config/models.json` 以定义onnx模型

## 使用方法

### 1. 克隆或下载项目

安装完成后，执行以下命令启动应用：

```bash
python labelImg.py
```

应用将启动图形用户界面，可以开始进行无人机图像标注工作。（现在只完成了yolo格式部分）

### 2. 自动标注
<<<<<<< HEAD
<p align="center">
=======
>>>>>>> 473c10e9487237302c6ecbbf8b688af5692c15d5
<img width="822" height="444" alt="image" src="https://github.com/user-attachments/assets/36e9dc68-294c-41b2-9319-1741e32ee240" /><br>
点击菜单“自动标注”选项进入自动标注界面，选择`Detection`或`Tracking`功能





### 3. 检测功能
<<<<<<< HEAD
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/fb963621-9fd8-4379-9ec8-9cbb755d52a8" /><br>
=======
<img width="716" height="510" alt="image" src="https://github.com/user-attachments/assets/fb963621-9fd8-4379-9ec8-9cbb755d52a8" /><br>
>>>>>>> 473c10e9487237302c6ecbbf8b688af5692c15d5
从`Detection Models`中选择已被定义的onnx模型，选择检测范围为单帧或文件夹内所有帧，并更改标注文件保存目录


### 4. 跟踪功能
<<<<<<< HEAD
<p align="center">
<img width="40%" alt="image" src="https://github.com/user-attachments/assets/a28ed4b4-b8c3-4c9a-9402-d85ac2a85d47" /><br>

自定义IOU阈值：当跟踪结果与原本图像中标注框IOU高于阈值时选择：<br>
保留原本标注框或采用跟踪结果作为标注框。<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/c687d660-83b9-4b14-aed7-5de5b778fecc" /><br>

选择`CSRT`或`ORTrack`作为跟踪器，选用`ORTrack`时可选是否用GPU加速。<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/d5d7f3db-845d-4fad-8ddb-7b83beec0a89" /><br>

选中任意目标作为起始目标，点击'Start Tracking'开始自动标注。<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/890deb62-7d29-466d-8d4d-59fc56edbc0d" /><br>

开始进行可视化跟踪，若发现跟踪错误则按下'Stop Tracking'停止。<br>
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/83325a9f-33ed-462d-bf21-77456a9a4cf8" /><br>
=======
<img width="427" height="533" alt="image" src="https://github.com/user-attachments/assets/a28ed4b4-b8c3-4c9a-9402-d85ac2a85d47" /><br>

自定义IOU阈值：当跟踪结果与原本图像中标注框IOU高于阈值时选择：<br>
保留原本标注框或采用跟踪结果作为标注框。<br>
<img width="505" height="312" alt="image" src="https://github.com/user-attachments/assets/c687d660-83b9-4b14-aed7-5de5b778fecc" /><br>

选择`CSRT`或`ORTrack`作为跟踪器，选用`ORTrack`时可选是否用GPU加速。<br>
<img width="405" height="215" alt="image" src="https://github.com/user-attachments/assets/d5d7f3db-845d-4fad-8ddb-7b83beec0a89" /><br>

选中任意目标作为起始目标，点击'Start Tracking'开始自动标注。<br>
<img width="647" height="514" alt="image" src="https://github.com/user-attachments/assets/890deb62-7d29-466d-8d4d-59fc56edbc0d" /><br>

开始进行可视化跟踪，若发现跟踪错误则按下'Stop Tracking'停止。<br>
<img width="645" height="512" alt="image" src="https://github.com/user-attachments/assets/83325a9f-33ed-462d-bf21-77456a9a4cf8" /><br>
>>>>>>> 473c10e9487237302c6ecbbf8b688af5692c15d5

进入保存界面，通过AD切换上下帧，停留在满意的跟踪帧时，按下Enter保存直到该帧的所有跟踪结果。<br>



## 功能特性

- 集成了基于目标检测与目标跟踪两种范式的半自动标注方法
- 加载onnx模型进行检测
- 集成CSRT与ORTrack两种跟踪方法进行无人机跟踪


## 许可证

基于labelImg项目开发

## 贡献

欢迎提交issue和pull request来帮助改进这个项目。

## Acknowledgements / Citation

This project utilizes the following open-source projects and algorithms:

*   **CSRT Tracker**: The implementation is based on the OpenCV library [1], which is licensed under the Apache 2.0 License. The underlying algorithm is from the paper:
    > A. Lukezic, et al. "Discriminative Correlation Filter with Channel and Spatial Reliability." *IJCV*, 2018. [4†L11-L12]

*   **ORTrack**: The implementation is based on the official repository [2] and the paper:
    > [Author Names], "Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking." *CVPR*, 2025. [1†L28]

[1] OpenCV: https://opencv.org/
[2] ORTrack GitHub Repository: https://github.com/wuyou3474/ORTrack
---
