# -*- coding: utf-8 -*-

import json
import os
from .onnx_detector import ONNXDetector


class ModelManager(object):
    """模型管理器 - 从JSON配置加载，管理ONNX模型缓存"""

    def __init__(self, config_path=None):
        """
        config_path: models.json 的路径
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'config', 'models.json'
            )

        self.config_path = config_path
        self.model_configs = self._load_config()
        self._detectors = {}

    def _load_config(self):
        """从models.json加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('models', [])
        except FileNotFoundError:
            raise FileNotFoundError(u'配置文件不存在: {}'.format(self.config_path))
        except json.JSONDecodeError:
            raise ValueError(u'配置文件格式错误: {}'.format(self.config_path))

    def get_model_list(self):
        """获取前端可用的模型列表"""
        return [
            {
                'id': cfg['id'],
                'name': cfg['name'],
                'description': cfg.get('description', '')
            }
            for cfg in self.model_configs
        ]

    def get_detector(self, model_id):
        """
        根据model_id获取或创建检测器
        相同model_id的模型只加载一次（缓存）
        """
        if model_id in self._detectors:
            return self._detectors[model_id]

        model_config = None
        for cfg in self.model_configs:
            if cfg['id'] == model_id:
                model_config = cfg
                break

        if not model_config:
            raise ValueError(u'未找到模型配置: {}'.format(model_id))

        detector = ONNXDetector(model_config)
        detector.load_model()

        self._detectors[model_id] = detector
        return detector

    def unload(self, model_id):
        """卸载模型释放资源"""
        if model_id in self._detectors:
            del self._detectors[model_id]

    def unload_all(self):
        """卸载所有模型"""
        self._detectors.clear()
