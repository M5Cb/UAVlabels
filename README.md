# UAVlabels

基于labelImg开发的无人机目标半自动化标注工具。

## 功能介绍

UAVlabels是一款专为无人机图像标注优化的工具，继承了labelImg的核心标注功能，并针对无人机目标检测场景进行了增强和优化。

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

从以下链接中获得用于无人机检测的onnx文件与ORTrack跟踪权重
链接: https://pan.baidu.com/s/1FHcaWI0PSyaesn59TnmBXw 提取码: qcuy 
下载后将模型文件放入autolabeling/models 中
编辑autolabeling/config/models.json 以定义onnx模型

## 使用方法

安装完成后，执行以下命令启动应用：

```bash
python labelImg.py
```

应用将启动图形用户界面，可以开始进行无人机图像标注工作。

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
