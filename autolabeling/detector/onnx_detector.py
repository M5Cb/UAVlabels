# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from typing import List, Tuple

try:
    import onnxruntime as ort
except ImportError:
    ort = None

Detection = Tuple[int, float, float, float, float, float]


class ONNXDetector(object):
    """通用ONNX检测器 - 支持任意ONNX模型"""

    def __init__(self, model_config: dict):
        """
        model_config: {
            'id': 'rtdetr_fpv',
            'name': 'RTDETR FPV Detection',
            'model_path': '/path/to/model.onnx',
            'input_size': [640, 640],
            'confidence_threshold': 0.5,
            'class_names': ['object']
        }
        """
        self.config = model_config
        self.session = None
        self.input_size = tuple(model_config['input_size'])
        self.confidence_threshold = model_config.get('confidence_threshold', 0.5)
        self.class_names = model_config.get('class_names', [])

    def load_model(self):
        """加载ONNX模型 - 通用加载逻辑"""
        if ort is None:
            raise RuntimeError('onnxruntime 未安装，请运行: pip install onnxruntime')

        try:
            self.session = ort.InferenceSession(self.config['model_path'])
            print(u'[OK] Model loaded: {}'.format(self.config['name']))
        except Exception as e:
            raise RuntimeError(u'Model loading failed: {}'.format(str(e)))

    def infer(self, image_path):
        """
        推理单张图片 - 通用流程

        1. 预处理图片
        2. 调用ONNX推理
        3. 后处理输出
        4. 返回YOLO格式结果
        """
        if not self.session:
            self.load_model()

        tensor, original_size = self._preprocess(image_path)
        model_output = self.session.run(
            output_names=None,
            input_feed={'images': tensor,
                       'orig_target_sizes': self._get_target_size()}
        )
        detections = self._postprocess(model_output, original_size)

        return detections

    def _preprocess(self, image_path):
        """图片预处理 - 缩放到模型要求尺寸"""
        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        img_resized = img.resize(self.input_size)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = img_array.transpose(2, 0, 1)
        img_array = img_array[np.newaxis, :, :, :]
        img_array = img_array / 255.0

        return img_array, original_size

    def _get_target_size(self):
        """获取ONNX模型期望的目标尺寸"""
        target_size = np.array([[self.input_size[0], self.input_size[1]]], dtype=np.int64)
        return target_size

    def _postprocess(self, model_output, original_size):
        """后处理模型输出 - 转换为YOLO格式"""

        if len(model_output) >= 3:
            labels, boxes, scores = model_output[0], model_output[1], model_output[2]
        else:
            labels, boxes, scores = model_output

        scale_x = original_size[0] / float(self.input_size[0])
        scale_y = original_size[1] / float(self.input_size[1])

        detections = []

        for batch_idx in range(labels.shape[0]):
            batch_scores = scores[batch_idx]
            if batch_scores is None:
                continue

            mask = batch_scores > self.confidence_threshold
            if not mask.any():
                continue

            batch_labels = labels[batch_idx][mask]
            batch_boxes = boxes[batch_idx][mask]
            batch_scores_filtered = batch_scores[mask]

            for box, label, score in zip(batch_boxes, batch_labels, batch_scores_filtered):
                x_min, y_min, x_max, y_max = box

                x_min_orig = float(x_min) * scale_x
                y_min_orig = float(y_min) * scale_y
                x_max_orig = float(x_max) * scale_x
                y_max_orig = float(y_max) * scale_y

                x_center = (x_min_orig + x_max_orig) / 2.0 / original_size[0]
                y_center = (y_min_orig + y_max_orig) / 2.0 / original_size[1]
                width = (x_max_orig - x_min_orig) / original_size[0]
                height = (y_max_orig - y_min_orig) / original_size[1]

                x_center = float(np.clip(x_center, 0, 1))
                y_center = float(np.clip(y_center, 0, 1))
                width = float(np.clip(width, 0, 1))
                height = float(np.clip(height, 0, 1))

                detections.append((
                    int(label),
                    x_center,
                    y_center,
                    width,
                    height,
                    float(score)
                ))

        return detections
