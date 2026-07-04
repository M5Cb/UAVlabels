# -*- coding: utf-8 -*-

from typing import Tuple


def calculate_iou(bbox1: Tuple[float, float, float, float],
                  bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个归一化边界框的 IoU（交并比）。
    输入格式均为 (x_center, y_center, width, height)，取值范围 [0,1]。
    """
    # 转换为左上角和右下角坐标
    x1_min = bbox1[0] - bbox1[2] / 2
    y1_min = bbox1[1] - bbox1[3] / 2
    x1_max = bbox1[0] + bbox1[2] / 2
    y1_max = bbox1[1] + bbox1[3] / 2

    x2_min = bbox2[0] - bbox2[2] / 2
    y2_min = bbox2[1] - bbox2[3] / 2
    x2_max = bbox2[0] + bbox2[2] / 2
    y2_max = bbox2[1] + bbox2[3] / 2

    # 计算交集坐标
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # 计算并集面积
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
