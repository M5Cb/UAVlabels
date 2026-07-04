# -*- coding: utf-8 -*-

import cv2
import os
import glob
from PIL import Image
from autolabeling.processor.annotation_loader import AnnotationLoader


class CSRTTracker:
    """CSRT目标跟踪器"""

    def __init__(self, class_label='object'):
        """
        参数:
            class_label: 跟踪对象的类别标签
        """
        self.class_label = class_label
        self.tracker = None
        self._annotations_loaded = False
        self._existing_annotations = {}
        self._image_dir_for_annotations = None

    def _get_image_list(self, folder):
        """获取文件夹内所有图片文件，按名称排序"""
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        files_set = set()
        for ext in exts:
            files_set.update(glob.glob(os.path.join(folder, ext)))
            files_set.update(glob.glob(os.path.join(folder, ext.upper())))

        files = list(files_set)
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
        return files

    def _ensure_annotations_loaded(self):
        """延迟加载注解：仅在首次需要时加载"""
        if not self._annotations_loaded and self._image_dir_for_annotations:
            try:
                loader = AnnotationLoader()
                self._existing_annotations = loader.load_annotations_from_folder(self._image_dir_for_annotations)
                print(f"[Tracking] Loaded annotations for {len(self._existing_annotations)} frames")
            except Exception as e:
                print(f"[Tracking] Failed to load annotations: {str(e)}")
                self._existing_annotations = {}
            self._annotations_loaded = True

    def track_folder(self, folder_path, start_frame_path, roi,
                     existing_annotations=None,
                     image_dir_for_annotations=None,
                     progress_callback=None):
        """
        跟踪文件夹中的图片序列

        参数:
            folder_path: 图片所在文件夹
            start_frame_path: 开始跟踪的帧路径
            roi: (x, y, w, h) 初始ROI
            existing_annotations: {frame_path: [(label, points, ...), ...]} (已废弃，为向后兼容保留)
            image_dir_for_annotations: 在需要时从该目录加载注解（延迟加载）
            progress_callback: 进度回调，签名为 callback(frame_path, current, total, bbox, success)
                              返回False表示用户中止

        返回: {
            'all_frames': [
                {
                    'frame_path': '...',
                    'frame_index': 0,
                    'shapes': [(label, points, ...), ...],
                    'success': True
                },
                ...
            ],
            'stopped_at_index': None  # 用户中止时的帧索引
        }
        """
        # 延迟加载：如果指定了 image_dir_for_annotations，则在首次使用时再加载
        if existing_annotations is None:
            existing_annotations = {}

        self._annotations_loaded = False
        self._image_dir_for_annotations = image_dir_for_annotations

        image_paths = self._get_image_list(folder_path)
        if not image_paths:
            return {'all_frames': [], 'stopped_at_index': None}

        start_idx = None
        for idx, path in enumerate(image_paths):
            if os.path.abspath(path) == os.path.abspath(start_frame_path):
                start_idx = idx
                break

        if start_idx is None:
            return {'all_frames': [], 'stopped_at_index': None}

        first_frame = cv2.imread(image_paths[start_idx])
        if first_frame is None:
            return {'all_frames': [], 'stopped_at_index': None}

        h, w = first_frame.shape[:2]

        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(first_frame, roi)

        all_frames = []

        bbox_display = roi
        x, y, w_box, h_box = [int(v) for v in bbox_display]
        shapes = [(self.class_label, [(x, y), (x + w_box, y), (x + w_box, y + h_box), (x, y + h_box)], None, None, False)]

        # 使用向后兼容的方式获取现有注解
        if existing_annotations:
            existing = existing_annotations.get(image_paths[start_idx], [])
        else:
            self._ensure_annotations_loaded()
            existing = self._existing_annotations.get(image_paths[start_idx], [])
        shapes.extend(existing)
        all_frames.append({
            'frame_path': image_paths[start_idx],
            'frame_index': start_idx,
            'shapes': shapes,
            'success': True,
            'raw_bbox': bbox_display
        })

        for idx, img_path in enumerate(image_paths[start_idx + 1:], start=start_idx + 1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            success, bbox = self.tracker.update(frame)

            shapes = None

            if success:
                x, y, w_box, h_box = [int(v) for v in bbox]
                shapes = [(self.class_label, [(x, y), (x + w_box, y), (x + w_box, y + h_box), (x, y + h_box)], None, None, False)]

                # 延迟加载注解
                if existing_annotations:
                    existing = existing_annotations.get(img_path, [])
                else:
                    self._ensure_annotations_loaded()
                    existing = self._existing_annotations.get(img_path, [])
                shapes.extend(existing)
            else:
                # 延迟加载注解
                if existing_annotations:
                    existing = existing_annotations.get(img_path, [])
                else:
                    self._ensure_annotations_loaded()
                    existing = self._existing_annotations.get(img_path, [])
                if existing:
                    shapes = existing

            frame_result = {
                'frame_path': img_path,
                'frame_index': idx,
                'shapes': shapes,
                'success': success,
                'raw_bbox': bbox if success else None
            }

            all_frames.append(frame_result)

            if progress_callback:
                should_continue = progress_callback(
                    img_path, idx, len(image_paths), bbox if success else None, success
                )
                if not should_continue:
                    return {
                        'all_frames': all_frames,
                        'stopped_at_index': idx,
                        'total_images': len(image_paths)
                    }

        return {
            'all_frames': all_frames,
            'stopped_at_index': None,
            'total_images': len(image_paths)
        }

    def track_single_frame(self, frame, existing_annotations=None):
        """
        跟踪单帧，用于实时显示

        返回: (success, bbox)
        """
        if self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)

        return success, bbox
