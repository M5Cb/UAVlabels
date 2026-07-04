# -*- coding: utf-8 -*-

import os
from PIL import Image
from .processor.box_utils import calculate_iou


class AnnotationAppender:
    """追加跟踪结果到现有标注文件"""

    @staticmethod
    def _load_existing_yolo_annotations(txt_path):
        """
        读取已有的YOLO格式标注文件

        参数:
            txt_path: 标注文件路径

        返回: 标注列表 [{'class_index': 0, 'x_center': 0.5, ...}, ...]
        """
        yolo_lines = []
        if not os.path.exists(txt_path):
            return yolo_lines

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                class_index = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                yolo_lines.append({
                                    'class_index': class_index,
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'w': w,
                                    'h': h
                                })
                            except ValueError:
                                print(u'Failed to parse YOLO line: {}'.format(line))
        except Exception as e:
            print(u'Failed to read existing YOLO file {}: {}'.format(txt_path, str(e)))

        return yolo_lines

    @staticmethod
    def _load_classes(save_dir):
        """
        加载classes.txt文件，获取标签名到索引的映射

        参数:
            save_dir: 标注文件保存目录

        返回: ({label: index}, [labels], classes_file_path)
        """
        classes_file = os.path.join(save_dir, 'classes.txt')
        label_to_index = {}
        labels = []

        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]
                    label_to_index = {label: idx for idx, label in enumerate(labels)}
            except Exception as e:
                print(u'Failed to load classes.txt: {}'.format(str(e)))

        return label_to_index, labels, classes_file

    @staticmethod
    def _save_classes(classes_file_path, labels):
        """
        保存类别列表到classes.txt文件

        参数:
            classes_file_path: classes.txt文件路径
            labels: 类别标签列表
        """
        try:
            os.makedirs(os.path.dirname(classes_file_path), exist_ok=True)
            with open(classes_file_path, 'w', encoding='utf-8') as f:
                for label in labels:
                    f.write(label + '\n')
        except Exception as e:
            print(u'Failed to save classes.txt: {}'.format(str(e)))

    @staticmethod
    def append_yolo_annotations(tracking_results, image_dir, save_dir=None, classes_file_path=None, iou_threshold=0.8, conflict_strategy='keep_existing'):
        """
        追加YOLO格式的跟踪标注到标注文件

        参数:
            tracking_results: 跟踪结果列表
                [{
                    'frame_path': '...',
                    'shapes': [(label, points, ...), ...],
                    'success': True
                }, ...]
            image_dir: 图片所在目录
            save_dir: 标注文件保存目录（如果为None则使用image_dir）
            classes_file_path: classes.txt文件路径，如果为None则自动查找
            iou_threshold: IoU阈值，跟踪框与已有标注IoU超过此值则应用冲突策略
            conflict_strategy: 冲突处理策略
                'keep_existing': IoU冲突时保留原有标注（跳过跟踪框）
                'use_tracking': IoU冲突时使用跟踪结果（覆盖原有标注）

        返回: 追加成功的文件数
        """
        if not tracking_results:
            return 0

        if save_dir is None:
            save_dir = image_dir

        label_to_index, labels, classes_file = AnnotationAppender._load_classes(save_dir)
        saved_count = 0

        for frame_result in tracking_results:
            if not frame_result.get('shapes'):
                continue

            frame_path = frame_result['frame_path']
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]
            txt_path = os.path.join(save_dir, frame_name + '.txt')

            try:
                image = Image.open(frame_path)
                img_width, img_height = image.size

                existing_yolo_lines = AnnotationAppender._load_existing_yolo_annotations(txt_path) if os.path.exists(txt_path) else []

                yolo_lines = []
                for shape in frame_result['shapes']:
                    label = shape[0]
                    points = shape[1] if len(shape) > 1 else []

                    if points and len(points) >= 4:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        x_center = (x_min + x_max) / 2.0 / img_width
                        y_center = (y_min + y_max) / 2.0 / img_height
                        w = (x_max - x_min) / img_width
                        h = (y_max - y_min) / img_height

                        current_bbox_norm = (x_center, y_center, w, h)

                        conflicting_indices = []
                        for idx, existing_line in enumerate(existing_yolo_lines):
                            existing_bbox_norm = (
                                existing_line['x_center'],
                                existing_line['y_center'],
                                existing_line['w'],
                                existing_line['h']
                            )
                            iou = calculate_iou(current_bbox_norm, existing_bbox_norm)
                            if iou > iou_threshold:
                                conflicting_indices.append(idx)

                        if conflicting_indices:
                            if conflict_strategy == 'keep_existing':
                                continue
                            elif conflict_strategy == 'use_tracking':
                                for idx in sorted(conflicting_indices, reverse=True):
                                    del existing_yolo_lines[idx]

                        if label not in label_to_index:
                            label_to_index[label] = len(labels)
                            labels.append(label)

                        class_index = label_to_index[label]

                        yolo_lines.append({
                            'class_index': class_index,
                            'x_center': x_center,
                            'y_center': y_center,
                            'w': w,
                            'h': h
                        })

                if yolo_lines or existing_yolo_lines:
                    if conflict_strategy == 'use_tracking':
                        final_yolo_lines = yolo_lines + existing_yolo_lines
                    else:
                        final_yolo_lines = existing_yolo_lines + yolo_lines
                    AnnotationAppender._write_yolo_annotations_only(
                        txt_path, final_yolo_lines
                    )
                    saved_count += 1

            except Exception as e:
                print(u'Failed to append annotations for {}: {}'.format(frame_path, str(e)))

        AnnotationAppender._save_classes(classes_file, labels)
        return saved_count

    @staticmethod
    def _write_yolo_annotations_only(txt_path, yolo_lines):
        """
        直接写入YOLO标注（覆盖已有内容）

        参数:
            txt_path: 标注文件路径
            yolo_lines: YOLO格式的标注行列表
                [{
                    'class_index': 0,
                    'x_center': 0.5,
                    'y_center': 0.5,
                    'w': 0.3,
                    'h': 0.4
                }, ...]
        """
        if not yolo_lines:
            return

        content_lines = []

        for line_data in yolo_lines:
            class_index = line_data['class_index']
            x_center = line_data['x_center']
            y_center = line_data['y_center']
            w = line_data['w']
            h = line_data['h']

            yolo_line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                class_index, x_center, y_center, w, h
            )
            content_lines.append(yolo_line)

        try:
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'w', encoding='utf-8') as f:
                if content_lines:
                    f.write('\n'.join(content_lines) + '\n')
        except Exception as e:
            print(u'Failed to write YOLO annotations to {}: {}'.format(txt_path, str(e)))

    @staticmethod
    def _write_yolo_annotations(txt_path, yolo_lines):
        """
        将YOLO标注追加到文件（追加而不覆盖）

        参数:
            txt_path: 标注文件路径
            yolo_lines: YOLO格式的标注行列表
                [{
                    'class_index': 0,
                    'x_center': 0.5,
                    'y_center': 0.5,
                    'w': 0.3,
                    'h': 0.4
                }, ...]
        """
        if not yolo_lines:
            return

        content_lines = []

        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    float(parts[1])
                                    float(parts[2])
                                    float(parts[3])
                                    float(parts[4])
                                    content_lines.append(line)
                                except ValueError:
                                    print(u'Skipping invalid YOLO line: {}'.format(line))
            except Exception as e:
                print(u'Failed to read existing YOLO file {}: {}'.format(txt_path, str(e)))

        for line_data in yolo_lines:
            class_index = line_data['class_index']
            x_center = line_data['x_center']
            y_center = line_data['y_center']
            w = line_data['w']
            h = line_data['h']

            yolo_line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                class_index, x_center, y_center, w, h
            )
            content_lines.append(yolo_line)

        try:
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'w', encoding='utf-8') as f:
                if content_lines:
                    f.write('\n'.join(content_lines) + '\n')
        except Exception as e:
            print(u'Failed to write YOLO annotations to {}: {}'.format(txt_path, str(e)))
