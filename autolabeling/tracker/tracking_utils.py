# -*- coding: utf-8 -*-
"""
追踪辅助函数 - 从AutoTracker中提取的通用工具
包含图像采样、预处理、坐标变换等功能
"""

import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class NestedTensor(object):
    """
    NestedTensor 容器 - 兼容AutoTracker的输入格式
    用于传递图像张量和对应的mask
    """
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask


def sample_target(im, target_bb, search_area_factor, output_sz=None):
    """
    从图像中提取目标区域，支持padding处理

    参数:
        im: 输入图像 (H, W, 3)
        target_bb: 目标框 [x, y, w, h]
        search_area_factor: 搜索区域相对于目标面积的倍数
        output_sz: 输出大小（若指定则进行resize）

    返回:
        img_crop: 提取的图像crop
        resize_factor: resize因子 (output_sz / crop_sz)
        att_mask: attention mask，标记有效像素（0=真实，1=padding）
    """
    if not isinstance(target_bb, list):
        if isinstance(target_bb, tuple):
            target_bb = list(target_bb)
        else:
            target_bb = target_bb.tolist()

    x, y, w, h = target_bb

    # 关键：根据目标面积计算搜索区域大小，而不是简单乘以倍数
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise ValueError('目标框过小')

    # 计算裁剪区域坐标（以目标中心为准）
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    # 计算需要padding的大小
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # 裁剪图像
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # 使用常数padding扩展到指定大小
    im_crop_padded = cv2.copyMakeBorder(
        im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=0
    )

    # 生成attention mask - 标记哪些像素是真实的，哪些是padding的
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W), dtype=np.uint8)
    end_x = -x2_pad if x2_pad > 0 else None
    end_y = -y2_pad if y2_pad > 0 else None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0  # 0表示有效像素

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz))
        att_mask = (att_mask > 127).astype(np.uint8)  # 转换回二值
    else:
        resize_factor = 1.0

    return im_crop_padded, resize_factor, att_mask


def normalize_image(img_tensor):
    """
    ImageNet标准归一化

    参数:
        img_tensor: 张量 (B, C, H, W) 范围[0, 1]或[0, 255]

    返回:
        img_norm: 归一化后的张量
    """
    # 如果是[0, 255]范围，先转到[0, 1]
    if img_tensor.max() > 1:
        img_tensor = img_tensor / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)

    return (img_tensor - mean) / std


def process_image(img_arr: np.ndarray, amask_arr: np.ndarray, device='cpu'):
    """
    处理图像和attention mask - 返回NestedTensor格式
    兼容AutoTracker的 DeviceAwarePreprocessor.process()

    参数:
        img_arr: 图像数组 (H, W, 3)
        amask_arr: Attention mask数组 (H, W)
        device: 设备

    返回:
        NestedTensor: 包含处理好的图像和mask
    """
    # 转换图像到张量
    img_tensor = torch.from_numpy(img_arr).to(device).float().permute((2, 0, 1)).unsqueeze(dim=0)
    img_tensor_norm = normalize_image(img_tensor)

    # 处理mask
    amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).to(device).unsqueeze(dim=0)

    return NestedTensor(img_tensor_norm, amask_tensor)


def hann2d(size, device='cpu'):
    """
    生成2D Hann窗口

    参数:
        size: 窗口大小 [H, W]
        device: 计算设备

    返回:
        hann_window: 2D Hann窗口张量
    """
    hann_1d = torch.hann_window(size[0], device=device)
    hann_2d = hann_1d[:, None] * hann_1d[None, :]
    return hann_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def map_box_back(pred_box, search_size, resize_factor, prev_state):
    """
    精确的坐标映射 - 从搜索特征空间映射回原始图像空间
    完全复制AutoTracker的 _map_box_back 方法

    参数:
        pred_box: 预测框 [cx, cy, w, h]
                 输入坐标是在搜索区域中的实际坐标（已通过 * search_size / resize_factor 缩放）
        search_size: 搜索大小（通常256）
        resize_factor: resize因子 - 从原始搜索区域大小到搜索特征图大小
        prev_state: 前一帧的目标状态 [x, y, w, h]

    返回:
        mapped_box: 映射到原始图像的框 [x, y, w, h]

    原理说明：
        搜索区域是以目标中心为准的正方形区域
        这个函数需要计算：
        1. 前一帧目标中心在原始图像中的位置
        2. 搜索区域在原始图像中的位置
        3. 预测框在搜索区域中的位置 → 转换到原始图像坐标系
    """
    # 计算前一帧目标中心（原始图像坐标）
    cx_prev = prev_state[0] + 0.5 * prev_state[2]
    cy_prev = prev_state[1] + 0.5 * prev_state[3]

    # 预测框坐标
    cx, cy, w, h = pred_box

    # 搜索区域的半边长（在原始图像坐标系中）
    # = 0.5 * search_size / resize_factor
    # 这是因为搜索特征图的大小是search_size，缩放因子是resize_factor
    half_side = 0.5 * search_size / resize_factor

    # 从搜索区域坐标转换回原始图像坐标
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)

    # 从中心坐标转换为左上角坐标
    x = cx_real - 0.5 * w
    y = cy_real - 0.5 * h

    return [x, y, w, h]


def clip_box(box, img_h, img_w, margin=10):
    """
    将框约束在图像范围内

    参数:
        box: [x, y, w, h]
        img_h, img_w: 图像高度和宽度
        margin: 保留的边界裕度

    返回:
        clipped_box: 约束后的框
    """
    x, y, w, h = box
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]
