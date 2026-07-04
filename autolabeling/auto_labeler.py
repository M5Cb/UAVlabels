# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image
from .detector.model_manager import ModelManager
from .processor.converter import DetectionToShapeConverter
from .tracker.csrt_tracker import CSRTTracker
from .processor.annotation_loader import AnnotationLoader

try:
    from libs.labelFile import LabelFile
    LabelFileError = Exception
except ImportError:
    LabelFile = None
    LabelFileError = Exception

_manager = None


def get_model_manager():
    """获取全局模型管理器（单例）"""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


class AutoLabeler(object):
    """自动标注主类 - UI直接调用这个"""

    def __init__(self, model_id):
        """
        model_id: 从models.json中选择的模型ID（如'rtdetr_fpv'）
        """
        manager = get_model_manager()
        self.detector = manager.get_detector(model_id)
        self.model_id = model_id

    def label_image(self, image_path):
        """
        标注单张图片

        返回: [(label, points, ...), ...]  # load_labels() 格式
        """
        detections = self.detector.infer(image_path)

        image_size = Image.open(image_path).size
        converter = DetectionToShapeConverter(
            class_names=self.detector.class_names,
            image_size=image_size
        )
        shapes = converter.convert(detections)

        return shapes

    def label_folder(self, folder_path, progress_callback=None, save_format=None, save_dir=None):
        """
        标注整个文件夹并保存到文件

        参数:
            folder_path: 图片文件夹路径
            progress_callback: 进度回调函数，签名为 progress_callback(current, total, image_path)
                              返回False表示用户点击了Cancel，应该中止处理
            save_format: 保存格式 ('pascal_voc', 'yolo', 'create_ml')，None则不保存
            save_dir: 保存标注文件的目录，None则与图片同目录

        返回: {
            'image1.jpg': {
                'shapes': [(label, points, ...), ...],
                'saved': True/False,
                'save_path': '...'
            },
            ...
        }
        """
        results = {}

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

        image_paths = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, '**/{}'.format(ext))
            image_paths.extend(glob.glob(pattern, recursive=True))

        total_count = len(image_paths)

        for current_idx, image_path in enumerate(image_paths, 1):
            should_continue = True
            if progress_callback:
                should_continue = progress_callback(current_idx, total_count, image_path)

            if not should_continue:
                print(u'Auto-labeling cancelled by user')
                break

            try:
                shapes = self.label_image(image_path)
                result_entry = {
                    'shapes': shapes,
                    'saved': False,
                    'save_path': None
                }

                if save_format:
                    save_path = self._save_annotation(
                        image_path, shapes, save_format, save_dir
                    )
                    result_entry['saved'] = True
                    result_entry['save_path'] = save_path

                results[image_path] = result_entry

            except Exception as e:
                print(u'Processing failed {}: {}'.format(image_path, str(e)))
                results[image_path] = {
                    'shapes': None,
                    'saved': False,
                    'save_path': None,
                    'error': str(e)
                }

        return results

    def _save_annotation(self, image_path, shapes, save_format, save_dir=None):
        """保存标注到文件"""
        if not LabelFile:
            print(u'Warning: LabelFile module not available')
            return None

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if save_dir is None:
            save_dir = os.path.dirname(image_path)

        os.makedirs(save_dir, exist_ok=True)

        label_file = LabelFile()

        def format_shape(s):
            return {
                'label': s[0],
                'points': s[1],
                'line_color': None,
                'fill_color': None,
                'difficult': s[4] if len(s) > 4 else False
            }

        formatted_shapes = [format_shape(shape) for shape in shapes]

        try:
            if save_format.lower() == 'yolo':
                annotation_path = os.path.join(save_dir, image_name + '.txt')
                label_file.save_yolo_format(
                    annotation_path, formatted_shapes, image_path,
                    Image.open(image_path), self.detector.class_names, None, None
                )
                self._save_classes_txt(save_dir, self.detector.class_names)
            elif save_format.lower() == 'pascal_voc':
                annotation_path = os.path.join(save_dir, image_name + '.xml')
                label_file.save_pascal_voc_format(
                    annotation_path, formatted_shapes, image_path,
                    Image.open(image_path), None, None
                )
            elif save_format.lower() == 'create_ml':
                annotation_path = os.path.join(save_dir, image_name + '.json')
                label_file.save_create_ml_format(
                    annotation_path, formatted_shapes, image_path,
                    Image.open(image_path), [], None, None
                )
            else:
                return None

            return annotation_path

        except Exception as e:
            print(u'Failed to save annotation {}: {}'.format(image_path, str(e)))
            return None

    @staticmethod
    def _save_classes_txt(save_dir, class_names):
        """保存类别列表到classes.txt文件

        参数:
            save_dir: 保存目录
            class_names: 类别名称列表
        """
        try:
            classes_file = os.path.join(save_dir, 'classes.txt')
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in class_names:
                    f.write(class_name + '\n')
        except Exception as e:
            print(u'Failed to save classes.txt: {}'.format(str(e)))

    def track_folder(self, folder_path, start_frame_path, roi, class_label,
                     progress_callback=None, save_format=None, save_dir=None):
        """
        跟踪图片序列并保存标注

        参数:
            folder_path: 图片所在文件夹
            start_frame_path: 开始跟踪的帧路径
            roi: (x, y, w, h) 初始ROI
            class_label: 跟踪对象的类别标签
            progress_callback: 进度回调
            save_format: 保存格式
            save_dir: 保存目录

        返回: {
            'image1.jpg': {
                'shapes': [(label, points, ...)],
                'saved': True/False,
                'save_path': '...',
                'frame_index': 0
            },
            ...
        }
        """
        results = {}

        loader = AnnotationLoader()
        existing_annotations = loader.load_annotations_from_folder(folder_path)

        tracker = CSRTTracker(class_label=class_label)
        tracking_results = tracker.track_folder(
            folder_path,
            start_frame_path,
            roi,
            existing_annotations=existing_annotations,
            progress_callback=progress_callback
        )

        if save_dir is None:
            save_dir = folder_path

        for frame_result in tracking_results['all_frames']:
            frame_path = frame_result['frame_path']
            frame_name = os.path.basename(frame_path)

            result_entry = {
                'shapes': frame_result['shapes'],
                'saved': False,
                'save_path': None,
                'frame_index': frame_result['frame_index'],
                'success': frame_result['success']
            }

            if save_format and frame_result['shapes']:
                try:
                    save_path = self._save_annotation(
                        frame_path, frame_result['shapes'], save_format, save_dir
                    )
                    result_entry['saved'] = True
                    result_entry['save_path'] = save_path
                except Exception as e:
                    print(u'Failed to save {}: {}'.format(frame_path, str(e)))

            results[frame_name] = result_entry

        if save_format and save_format.lower() == 'yolo':
            self._save_classes_txt(save_dir, self.detector.class_names)

        return results
