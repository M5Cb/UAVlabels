"""
Model builder for AutoTracker
"""
import os
import sys
import torch
from torch import nn

# Use internal lib instead of external lib
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from lib.models.layers.head import build_box_head
from lib.models.ortrack.ortrack import ORTrack
from lib.models.ortrack.vision_transformer import vit_tiny_patch16_224, vit_tiny_distilled_patch16_224
from lib.models.ortrack.deit import deit_tiny_patch16_224, deit_tiny_patch16_224_distill
from lib.models.ortrack.eva import eva02_tiny_patch14_224, eva02_tiny_patch14_224_distill


def build_model(cfg):
    """
    Build ORTrack model based on configuration

    Args:
        cfg: Configuration object with MODEL settings

    Returns:
        model: ORTrack model instance
    """
    backbone_type = cfg.MODEL.BACKBONE.TYPE

    if backbone_type == 'deit_tiny_patch16_224':
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif backbone_type == 'deit_tiny_distilled_patch16_224':
        backbone = deit_tiny_patch16_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif backbone_type == 'vit_tiny_patch16_224':
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif backbone_type == 'vit_tiny_distilled_patch16_224':
        backbone = vit_tiny_distilled_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif backbone_type == 'eva02_tiny_patch14_224':
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif backbone_type == 'eva02_tiny_distilled_patch14_224':
        backbone = eva02_tiny_patch14_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError(f'Backbone type {backbone_type} not implemented')

    box_head = build_box_head(cfg, hidden_dim)

    model = ORTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # Call finetune_track if the backbone supports it
    if hasattr(backbone, 'finetune_track'):
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    return model


def load_checkpoint(model, checkpoint_path):
    """
    Load checkpoint into model with compatibility for missing modules

    Args:
        model: ORTrack model instance
        checkpoint_path: Path to checkpoint file

    Returns:
        model: Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except ModuleNotFoundError as e:
        # Handle missing modules like lib.train.admin
        import pickle
        import sys

        class UnpicklerIgnoreMissing(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    print(f'[WARNING] Ignoring missing module: {module}.{name}')
                    return None

        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu', pickle_module=UnpicklerIgnoreMissing, weights_only=False)

    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f'[WARNING] Missing keys: {missing_keys[:3]}...')
    if unexpected_keys:
        print(f'[WARNING] Unexpected keys: {unexpected_keys[:3]}...')

    return model
