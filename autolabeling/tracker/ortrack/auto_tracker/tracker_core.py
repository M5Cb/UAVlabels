"""
Core tracking logic for AutoTracker
"""
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Use internal lib
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.utils.hann import hann2d

from preprocessor import DeviceAwarePreprocessor

from model_builder import build_model, load_checkpoint
from config import AutoTrackerConfig


class AutoTracker:
    """
    Standalone ORTrack-based tracker for automatic annotation
    """

    def __init__(self, checkpoint_path, cfg=None, use_cuda=True):
        """
        Initialize AutoTracker

        Args:
            checkpoint_path: Path to checkpoint file
            cfg: AutoTrackerConfig instance, uses default if None
            use_cuda: Whether to use CUDA
        """
        if cfg is None:
            cfg = AutoTrackerConfig()

        self.cfg = cfg.cfg
        self.params = cfg
        self.checkpoint_path = checkpoint_path
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Determine device
        if self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Build and load model
        print('[AutoTracker] Building model...')
        self.network = build_model(self.cfg)
        print('[AutoTracker] Loading checkpoint from:', checkpoint_path)
        self.network = load_checkpoint(self.network, checkpoint_path)

        self.network = self.network.to(self.device)
        self.network.eval()

        # Initialize preprocessor with device
        self.preprocessor = DeviceAwarePreprocessor(device=self.device)

        # Feature size calculation
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.feat_template_sz = self.cfg.TEST.TEMPLATE_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # Hann window for motion constraint
        hann_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)
        self.output_window = hann_window.to(self.device)

        # Tracker state
        self.state = None
        self.z_dict = None
        self.frame_id = 0

    def initialize(self, image, init_bbox):
        """
        Initialize tracker with first frame and bounding box

        Args:
            image: Initial image (H, W, 3) in RGB format
            init_bbox: Initial bounding box [x, y, w, h]
        """
        # Get template
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, init_bbox, self.params.template_factor,
            output_sz=self.params.template_size
        )
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict = template

        # Handle CE (Candidate Elimination) if enabled
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            self.box_mask_z = self._generate_mask_cond(init_bbox, resize_factor, template.tensors.device)
        else:
            self.box_mask_z = None

        # Save initial state
        self.state = init_bbox
        self.frame_id = 0

    def track(self, image):
        """
        Track object in frame

        Args:
            image: Current frame image (H, W, 3) in RGB format

        Returns:
            bbox: Predicted bounding box [x, y, w, h]
        """
        H, W, _ = image.shape
        self.frame_id += 1

        # Get search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # Run inference
        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict.tensors,
                search=search.tensors,
                is_distill=self.cfg.MODEL.IS_DISTILL
            )

        # Process output
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map']
        )
        pred_boxes = pred_boxes.view(-1, 4)

        # Get final box (average of all predictions)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()

        # Map box back to original image
        self.state = clip_box(self._map_box_back(pred_box, resize_factor), H, W, margin=10)

        return self.state

    def _map_box_back(self, pred_box, resize_factor):
        """
        Map predicted box from search region back to original image coordinates

        Args:
            pred_box: Predicted box in search region [cx, cy, w, h]
            resize_factor: Resize factor from original to search

        Returns:
            box: Mapped box in original image [x, y, w, h]
        """
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _generate_mask_cond(self, bbox, resize_factor, device):
        """
        Generate mask condition for CE (Candidate Elimination)

        Args:
            bbox: Bounding box [x, y, w, h]
            resize_factor: Resize factor
            device: Torch device

        Returns:
            mask: Generated mask condition
        """
        template_bbox = self._transform_bbox_to_crop(
            bbox, resize_factor, device, crop_type='template'
        ).squeeze(1)
        return generate_mask_cond(self.cfg, 1, device, template_bbox)

    def _transform_bbox_to_crop(self, box_in, resize_factor, device, box_extract=None, crop_type='template'):
        """
        Transform bounding box to crop coordinates

        Args:
            box_in: Bounding box [x, y, w, h]
            resize_factor: Resize factor
            device: Torch device
            box_extract: Box to extract, uses box_in if None
            crop_type: 'template' or 'search'

        Returns:
            bbox: Transformed bounding box tensor [1, 1, 4]
        """
        from lib.train.data.processing_utils import transform_image_to_crop

        if crop_type == 'template':
            crop_sz = torch.Tensor([self.params.template_size, self.params.template_size])
        elif crop_type == 'search':
            crop_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)

        template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox
