# -*- coding: utf-8 -*-
"""
ORTrack追踪器包装器 - 与CSRTTracker接口兼容
改进版本：完全集成AutoTracker的实现
  - 使用面积感知的搜索区域提取
  - NestedTensor格式支持
  - 正确的坐标缩放处理
  - Hann窗口运动约束
"""

import os
import sys
import glob
import cv2
import torch
import numpy as np
import math
from .tracking_utils import sample_target, normalize_image, hann2d, clip_box, NestedTensor, process_image, map_box_back
from autolabeling.processor.annotation_loader import AnnotationLoader


class ORTrackWrapper:
    """
    ORTrack目标跟踪器包装器

    与CSRTTracker接口保持一致，支持相同的track_folder()和track_single_frame()方法
    """

    def __init__(self, class_label='object', checkpoint_path=None, use_cuda=True, device_id=0):
        """
        初始化ORTrack追踪器

        参数:
            class_label: 跟踪对象的类别标签
            checkpoint_path: ORTrack模型权重路径
                           若为None，则尝试从default location加载
            use_cuda: 是否使用GPU加速
            device_id: GPU设备ID（当use_cuda=True时）

        异常:
            FileNotFoundError: 模型权重文件不存在
            RuntimeError: GPU不可用但use_cuda=True
        """
        self.class_label = class_label
        self.checkpoint_path = checkpoint_path
        self.use_cuda = use_cuda
        self.device_id = device_id

        # 设备配置
        self.device = self._setup_device()

        # 追踪参数（来自AutoTracker）
        self.template_factor = 2.0  # 模板搜索区域倍数
        self.search_factor = 5.0    # 搜索区域倍数
        self.template_size = 128    # 模板尺寸
        self.search_size = 256      # 搜索区域尺寸

        # 模型和配置
        self.model = None
        self.cfg = None
        self.template_box = None
        self.last_frame = None
        self.z_patch = None        # 保存的模板patch
        self.z_resize_factor = None # 模板resize因子

        # 初始化Hann窗口用于运动约束
        # 特征图大小 = 搜索大小 / backbone stride（通常为16）
        feat_sz = self.search_size // 16
        self.output_window = hann2d([feat_sz, feat_sz], device=self.device)

        # 延迟加载相关变量
        self._annotations_loaded = False
        self._existing_annotations = {}
        self._image_dir_for_annotations = None

        # 加载模型
        self._load_model()

    def _setup_device(self):
        """
        设置计算设备

        返回:
            torch.device: CPU或CUDA设备
        """
        if self.use_cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.device_id}')
            print(f"[ORTrack] Using GPU: {torch.cuda.get_device_name(self.device_id)}")
        else:
            device = torch.device('cpu')
            if self.use_cuda:
                print("[ORTrack] GPU not available, falling back to CPU")
            else:
                print("[ORTrack] Using CPU")
        return device

    def _load_model(self):
        """
        加载ORTrack模型和配置

        模型加载流程:
        1. 确定checkpoint路径
        2. 从ortrack模块导入配置和模型
        3. 加载预训练权重
        4. 将模型移到指定设备
        """
        # 确定checkpoint路径
        if self.checkpoint_path is None:
            # 尝试从默认位置加载
            default_path = os.path.join(
                os.path.dirname(__file__),
                './ortrack/pretrained_models/ortrack_default.pth'
            )
            if os.path.exists(default_path):
                self.checkpoint_path = default_path
            else:
                raise FileNotFoundError(
                    f"ORTrack checkpoint not found. "
                    f"Please provide checkpoint_path or place model at: {default_path}"
                )

        # 验证文件存在
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # 导入ortrack模块
        try:
            import warnings
            ortrack_dir = os.path.join(os.path.dirname(__file__), './ortrack')
            if ortrack_dir not in sys.path:
                sys.path.insert(0, ortrack_dir)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*Overwriting.*in registry.*')
                warnings.filterwarnings('ignore', category=FutureWarning, message='.*Importing from.*deprecated.*')
                from auto_tracker.config import AutoTrackerConfig
                from auto_tracker.model_builder import build_model, load_checkpoint

            # 加载配置
            self.cfg = AutoTrackerConfig().cfg

            # 构建模型
            self.model = build_model(self.cfg)
            self.model = self.model.to(self.device)
            self.model.eval()

            # 加载权重
            self.model = load_checkpoint(self.model, self.checkpoint_path)

            print(f"[ORTrack] Model loaded successfully from: {self.checkpoint_path}")

        except ImportError as e:
            raise RuntimeError(f"Failed to import ORTrack modules: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ORTrack model: {str(e)}")

    def _get_image_list(self, folder):
        """
        获取文件夹内所有图片文件，按名称排序

        支持的格式: jpg, jpeg, png, bmp, tif, tiff
        """
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        files_set = set()
        for ext in exts:
            files_set.update(glob.glob(os.path.join(folder, ext)))
            files_set.update(glob.glob(os.path.join(folder, ext.upper())))

        files = list(files_set)
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
        return files

    def _extract_template(self, frame, roi):
        """
        从帧中提取模板区域 - 完全兼容AutoTracker

        参数:
            frame: OpenCV图像 (H, W, 3) RGB格式
            roi: 初始ROI (x, y, w, h)

        返回:
            template: NestedTensor 格式 (兼容模型输入)
            resize_factor: resize因子
        """
        # 使用面积感知的采样方法
        z_patch, resize_factor, z_att_mask = sample_target(
            frame, roi, self.template_factor, output_sz=self.template_size
        )

        # 转换为NestedTensor格式 - 完全兼容AutoTracker
        template = process_image(z_patch, z_att_mask, device=self.device)

        return template, resize_factor

    def _extract_search_region(self, frame, prev_bbox):
        """
        从帧中提取搜索区域 - 完全兼容AutoTracker

        参数:
            frame: OpenCV图像 (H, W, 3) RGB格式
            prev_bbox: 上一帧的bbox (x, y, w, h)

        返回:
            search: NestedTensor 格式 (兼容模型输入)
            resize_factor: resize因子
        """
        # 使用面积感知的采样方法
        x_patch, resize_factor, x_att_mask = sample_target(
            frame, prev_bbox, self.search_factor, output_sz=self.search_size
        )

        # 转换为NestedTensor格式 - 完全兼容AutoTracker
        search = process_image(x_patch, x_att_mask, device=self.device)

        return search, resize_factor

    def _update_bbox(self, model_output, x_resize_factor, current_bbox, frame_h, frame_w):
        """
        从模型输出更新bbox - 完全复制AutoTracker的tracker_core.py第138-149行逻辑

        参数:
            model_output: 模型输出字典 (包含 score_map, size_map, offset_map)
            x_resize_factor: 搜索区域的resize因子
            current_bbox: 当前bbox (x, y, w, h)
            frame_h, frame_w: 帧高度和宽度

        返回:
            new_bbox: 更新后的bbox (x, y, w, h)
            confidence: 追踪置信度 [0, 1]

        ⚠️ 关键修复：实现与AutoTracker完全相同的坐标处理流程
        """
        # 第1步：获取预测结果（完全按照AutoTracker第138-142行）
        pred_score_map = model_output.get('score_map')
        size_map = model_output.get('size_map')
        offset_map = model_output.get('offset_map')

        if pred_score_map is None:
            return current_bbox, 0.0

        # 第2步：应用Hann窗口约束到响应图
        response = self.output_window * pred_score_map

        # 第3步：通过box_head计算最终预测框 ⚠️ 这是关键！
        # 注意：self.model.box_head 可能不存在，但输出中可能已经有 pred_boxes
        pred_boxes = model_output.get('pred_boxes')

        if pred_boxes is None:
            # 如果模型输出中没有pred_boxes，尝试从box_head计算
            # 这需要 self.model 有 box_head 属性
            if hasattr(self.model, 'box_head'):
                try:
                    pred_boxes = self.model.box_head.cal_bbox(response, size_map, offset_map)
                except:
                    print("[ORTrack] Warning: box_head.cal_bbox() failed, using pred_boxes from output")
                    return current_bbox, 0.0
            else:
                print("[ORTrack] Warning: No pred_boxes in output and no box_head available")
                return current_bbox, 0.0

        # 第4步：reshape预测框
        pred_boxes = pred_boxes.view(-1, 4)

        # 第5步：取均值并进行关键的缩放！⚠️ 这是最重要的修复！
        # AutoTracker第146行：pred_box = (pred_boxes.mean(dim=0) * search_size / resize_factor).tolist()
        # 这一步将坐标从搜索特征图空间转换到实际搜索区域大小
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / x_resize_factor).cpu().detach().numpy()
        pred_box = pred_box.tolist()

        # 第6步：从搜索区域映射回原始图像
        new_bbox = map_box_back(pred_box, self.search_size, x_resize_factor, current_bbox)

        # 第7步：约束在图像范围内
        new_bbox = clip_box(new_bbox, frame_h, frame_w, margin=10)

        # 计算置信度
        confidence = float(pred_score_map.max().cpu().detach().numpy())
        confidence = max(0.0, min(1.0, confidence))

        # 转换为整数坐标
        new_bbox = [int(round(v)) for v in new_bbox]

        return new_bbox, confidence

    def _ensure_annotations_loaded(self):
        """延迟加载注解：仅在首次需要时加载"""
        if not self._annotations_loaded and self._image_dir_for_annotations:
            try:
                loader = AnnotationLoader()
                self._existing_annotations = loader.load_annotations_from_folder(self._image_dir_for_annotations)
                print(f"[ORTrack] Loaded annotations for {len(self._existing_annotations)} frames")
            except Exception as e:
                print(f"[ORTrack] Failed to load annotations: {str(e)}")
                self._existing_annotations = {}
            self._annotations_loaded = True

    def track_folder(self, folder_path, start_frame_path, roi,
                     existing_annotations=None,
                     image_dir_for_annotations=None,
                     progress_callback=None):
        """
        跟踪文件夹中的图片序列 - 改进版本集成AutoTracker技术

        参数:
            folder_path: 图片所在文件夹
            start_frame_path: 开始跟踪的帧路径
            roi: (x, y, w, h) 初始ROI
            existing_annotations: {frame_path: [(label, points, ...), ...]} (已废弃，为向后兼容保留)
            image_dir_for_annotations: 在需要时从该目录加载注解（延迟加载）
            progress_callback: 进度回调函数
                              签名: callback(frame_path, current, total, bbox, success)
                              返回False表示用户中止

        返回: 与CSRTTracker完全相同的格式
        {
            'all_frames': [
                {
                    'frame_path': '...',
                    'frame_index': 0,
                    'shapes': [(label, points, ...), ...],
                    'success': True,
                    'raw_bbox': (x, y, w, h),
                    'confidence': 0.95
                },
                ...
            ],
            'stopped_at_index': None,
            'total_images': total_count
        }
        """
        # 延迟加载：如果指定了 image_dir_for_annotations，则在首次使用时再加载
        if existing_annotations is None:
            existing_annotations = {}

        self._annotations_loaded = False
        self._image_dir_for_annotations = image_dir_for_annotations

        # 获取有序的图片列表
        image_paths = self._get_image_list(folder_path)
        if not image_paths:
            return {'all_frames': [], 'stopped_at_index': None}

        print(f"[ORTrack] 开始追踪: ROI={roi}, 总图片数={len(image_paths)}")

        # 找到起始帧索引
        start_idx = None
        for idx, path in enumerate(image_paths):
            if os.path.abspath(path) == os.path.abspath(start_frame_path):
                start_idx = idx
                break

        if start_idx is None:
            return {'all_frames': [], 'stopped_at_index': None}

        # 读取第一帧
        first_frame = cv2.imread(image_paths[start_idx])
        if first_frame is None:
            return {'all_frames': [], 'stopped_at_index': None}

        # 转换BGR到RGB
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        # 提取模板 - 使用新的方法
        template, z_resize_factor = self._extract_template(first_frame_rgb, roi)
        self.z_patch = template
        self.z_resize_factor = z_resize_factor
        self.template_box = roi
        self.last_frame = first_frame_rgb

        all_frames = []

        # 处理第一帧
        x, y, w_box, h_box = [int(v) for v in roi]
        shapes = [(self.class_label, [(x, y), (x + w_box, y), (x + w_box, y + h_box), (x, y + h_box)],
                   None, None, False)]

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
            'raw_bbox': roi,
            'confidence': 1.0
        })

        current_bbox = roi

        # 追踪后续帧
        with torch.no_grad():
            for idx, img_path in enumerate(image_paths[start_idx + 1:], start=start_idx + 1):
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_h, frame_w = frame_rgb.shape[:2]

                # 提取搜索区域 - 使用新的方法
                try:
                    search, x_resize_factor = self._extract_search_region(frame_rgb, current_bbox)

                    # 前向推理 - 使用NestedTensor格式
                    with torch.no_grad():
                        output = self.model(template.tensors, search.tensors)

                    # 更新bbox - 使用精确的映射
                    new_bbox, confidence = self._update_bbox(output, x_resize_factor, current_bbox, frame_h, frame_w)
                    success = confidence > 0.1

                    print(f"[ORTrack] 帧{idx}: bbox={new_bbox}, 置信度={confidence:.3f}")

                except Exception as e:
                    print(f"[ORTrack] 帧{idx}处理错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    success = False
                    new_bbox = current_bbox
                    confidence = 0.0

                # 生成标注
                shapes = None
                if success:
                    x, y, w_box, h_box = [int(v) for v in new_bbox]
                    shapes = [(self.class_label, [(x, y), (x + w_box, y), (x + w_box, y + h_box), (x, y + h_box)],
                               None, None, False)]

                    # 延迟加载注解
                    if existing_annotations:
                        existing = existing_annotations.get(img_path, [])
                    else:
                        self._ensure_annotations_loaded()
                        existing = self._existing_annotations.get(img_path, [])
                    shapes.extend(existing)
                    current_bbox = new_bbox
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
                    'raw_bbox': new_bbox if success else None,
                    'confidence': confidence
                }

                all_frames.append(frame_result)

                # 进度回调
                if progress_callback:
                    should_continue = progress_callback(
                        img_path, idx, len(image_paths),
                        new_bbox if success else None, success
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
        跟踪单帧 - 改进版本

        参数:
            frame: OpenCV图像
            existing_annotations: 已有标注（未使用，保持接口一致性）

        返回:
            (success, bbox) - 追踪是否成功，边界框坐标
        """
        if self.z_patch is None or self.template_box is None:
            return False, None

        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_rgb.shape[:2]

        # 提取搜索区域
        try:
            search, x_resize_factor = self._extract_search_region(frame_rgb, self.template_box)

            # 前向推理 - 使用NestedTensor格式
            with torch.no_grad():
                output = self.model(self.z_patch.tensors, search.tensors)

            # 更新bbox
            new_bbox, confidence = self._update_bbox(output, x_resize_factor, self.template_box, frame_h, frame_w)
            success = confidence > 0.1

            if success:
                self.template_box = new_bbox

            return success, new_bbox if success else None

        except Exception as e:
            print(f"[ORTrack] 单帧追踪错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None

    def cleanup(self):
        """
        清理资源，释放GPU内存
        """
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available() and self.use_cuda:
            torch.cuda.empty_cache()

        self.template_box = None
        self.last_frame = None
