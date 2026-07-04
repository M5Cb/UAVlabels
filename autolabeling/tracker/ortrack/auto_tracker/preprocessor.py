"""
Device-aware preprocessor for AutoTracker
支持CPU和GPU的预处理器
"""
import os
import sys
import torch
import numpy as np

# Use internal lib
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from lib.utils.misc import NestedTensor


class DeviceAwarePreprocessor:
    """
    Preprocessor that works on CPU or GPU
    支持CPU和GPU的预处理器
    """

    def __init__(self, device='cpu'):
        """
        Initialize preprocessor

        Args:
            device: 'cpu' or 'cuda' or torch.device
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).to(self.device)

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """
        Process image and attention mask

        Args:
            img_arr: Image array (H, W, 3)
            amask_arr: Attention mask array (H, W)

        Returns:
            NestedTensor with processed image and mask
        """
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).to(self.device).float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std

        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).to(self.device).unsqueeze(dim=0)

        return NestedTensor(img_tensor_norm, amask_tensor)
