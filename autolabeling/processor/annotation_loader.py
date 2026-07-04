# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image


class AnnotationLoader:
    """加载现有的标注文件"""

    @staticmethod
    def load_annotations_from_folder(folder_path):
        """
        从文件夹加载所有现有标注。
        返回格式: {
            'image1.jpg': [(label, points, ...), ...],
            'image2.jpg': [(label, points, ...), ...],
            ...
        }
        """
        annotations = {}

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, f'**/{ext}')
            image_files.extend(glob.glob(pattern, recursive=True))

        for image_path in image_files:
            image_name = os.path.basename(image_path)

            txt_path = os.path.splitext(image_path)[0] + '.txt'
            xml_path = os.path.splitext(image_path)[0] + '.xml'
            json_path = os.path.splitext(image_path)[0] + '.json'

            shapes = None
            if os.path.exists(txt_path):
                shapes = AnnotationLoader._load_yolo_txt(txt_path, image_path)
            elif os.path.exists(xml_path):
                shapes = AnnotationLoader._load_pascal_voc_xml(xml_path, image_path)
            elif os.path.exists(json_path):
                shapes = AnnotationLoader._load_create_ml_json(json_path, image_path)

            if shapes:
                annotations[image_path] = shapes

        return annotations

    @staticmethod
    def _load_yolo_txt(txt_path, image_path):
        """加载YOLO格式的标注"""
        try:
            image = Image.open(image_path)
            img_w, img_h = image.size

            shapes = []
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    x_min = int((x_center - width / 2.0) * img_w)
                    y_min = int((y_center - height / 2.0) * img_h)
                    x_max = int((x_center + width / 2.0) * img_w)
                    y_max = int((y_center + height / 2.0) * img_h)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_w, x_max)
                    y_max = min(img_h, y_max)

                    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                    label = f"class_{class_id}"
                    shapes.append((label, points, None, None, False))

            return shapes
        except Exception as e:
            print(f"加载YOLO标注失败 {txt_path}: {str(e)}")
            return None

    @staticmethod
    def _load_pascal_voc_xml(xml_path, image_path):
        """加载Pascal VOC格式的标注"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()

            shapes = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                bndbox = obj.find('bndbox')
                x_min = int(bndbox.find('xmin').text)
                y_min = int(bndbox.find('ymin').text)
                x_max = int(bndbox.find('xmax').text)
                y_max = int(bndbox.find('ymax').text)

                points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                shapes.append((label, points, None, None, False))

            return shapes
        except Exception as e:
            print(f"加载Pascal VOC标注失败 {xml_path}: {str(e)}")
            return None

    @staticmethod
    def _load_create_ml_json(json_path, image_path):
        """加载Create ML格式的标注"""
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image = Image.open(image_path)
            img_w, img_h = image.size

            shapes = []
            annotations = data.get('annotations', [])
            for ann in annotations:
                label = ann.get('label', 'unknown')
                coordinates = ann.get('coordinates', [])
                if not coordinates:
                    continue

                xs = [c.get('x', 0) for c in coordinates]
                ys = [c.get('y', 0) for c in coordinates]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                shapes.append((label, points, None, None, False))

            return shapes
        except Exception as e:
            print(f"加载Create ML标注失败 {json_path}: {str(e)}")
            return None
