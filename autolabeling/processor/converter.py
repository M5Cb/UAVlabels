# -*- coding: utf-8 -*-

from typing import List, Tuple

Detection = Tuple[int, float, float, float, float, float]


class DetectionToShapeConverter(object):
    """将检测结果转换为Shape对象"""

    def __init__(self, class_names, image_size):
        """
        class_names: ['person', 'car', ...]
        image_size: (width, height)
        """
        self.class_names = class_names
        self.image_size = image_size

    def convert(self, detections):
        """
        将YOLO格式转换为 load_labels() 期望的格式

        返回: [(label, points, line_color, fill_color, difficult), ...]
        其中 points = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        shapes = []
        for class_id, x_c, y_c, w, h, conf in detections:
            class_id = int(class_id)
            if class_id < len(self.class_names):
                label = self.class_names[class_id]
            else:
                label = 'class_{}'.format(class_id)

            x_min = int((x_c - w / 2.0) * self.image_size[0])
            y_min = int((y_c - h / 2.0) * self.image_size[1])
            x_max = int((x_c + w / 2.0) * self.image_size[0])
            y_max = int((y_c + h / 2.0) * self.image_size[1])

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.image_size[0], x_max)
            y_max = min(self.image_size[1], y_max)

            points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

            shapes.append((
                label,
                points,
                None,
                None,
                False
            ))

        return shapes
