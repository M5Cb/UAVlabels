# -*- coding: utf-8 -*-
"""
追踪器工厂 - 支持多种追踪器类型的创建和管理
"""

from .csrt_tracker import CSRTTracker
from .ortrack_wrapper import ORTrackWrapper


class TrackerFactory:
    """追踪器工厂类"""

    AVAILABLE_TRACKERS = {
        'csrt': {
            'name': 'CSRT (Lightweight)',
            'class': CSRTTracker,
            'require_model_path': False,
            'require_gpu': False,
            'description': 'Fast correlation filter tracker, works on CPU'
        },
        'ortrack': {
            'name': 'ORTrack (High-Precision)',
            'class': ORTrackWrapper,
            'require_model_path': True,
            'require_gpu': False,
            'description': 'Vision Transformer based tracker, high accuracy'
        }
    }

    @staticmethod
    def get_available_trackers():
        """获取所有可用的追踪器列表"""
        return TrackerFactory.AVAILABLE_TRACKERS

    @staticmethod
    def create_tracker(tracker_type='csrt', class_label='object', **kwargs):
        """
        创建追踪器实例

        参数:
            tracker_type: 'csrt' 或 'ortrack'
            class_label: 标注类别标签
            **kwargs: 传递给具体追踪器的额外参数
                - ortrack特有: checkpoint_path, use_cuda, device_id
                - csrt无特殊参数

        返回:
            追踪器实例（CSRTTracker或ORTrackWrapper）

        异常:
            ValueError: 不支持的追踪器类型
            FileNotFoundError: 模型权重文件不存在（ORTrack）
        """
        if tracker_type not in TrackerFactory.AVAILABLE_TRACKERS:
            raise ValueError(f"Unknown tracker type: {tracker_type}. "
                           f"Available: {list(TrackerFactory.AVAILABLE_TRACKERS.keys())}")

        tracker_info = TrackerFactory.AVAILABLE_TRACKERS[tracker_type]
        TrackerClass = tracker_info['class']

        try:
            if tracker_type == 'csrt':
                return TrackerClass(class_label=class_label)
            elif tracker_type == 'ortrack':
                return TrackerClass(
                    class_label=class_label,
                    checkpoint_path=kwargs.get('checkpoint_path'),
                    use_cuda=kwargs.get('use_cuda', True),
                    device_id=kwargs.get('device_id', 0)
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create {tracker_type} tracker: {str(e)}")

    @staticmethod
    def validate_tracker_config(tracker_type, config):
        """
        验证追踪器配置是否有效

        参数:
            tracker_type: 追踪器类型
            config: 配置字典

        返回:
            (valid: bool, message: str)
        """
        if tracker_type not in TrackerFactory.AVAILABLE_TRACKERS:
            return False, f"Unknown tracker type: {tracker_type}"

        tracker_info = TrackerFactory.AVAILABLE_TRACKERS[tracker_type]

        if tracker_info['require_model_path']:
            checkpoint_path = config.get('checkpoint_path')
            if not checkpoint_path:
                return False, "ORTrack requires checkpoint_path"

            import os
            if not os.path.exists(checkpoint_path):
                return False, f"Checkpoint file not found: {checkpoint_path}"

        return True, "Config is valid"
