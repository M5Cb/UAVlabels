"""
Configuration for AutoTracker module
"""
from easydict import EasyDict as edict


class AutoTrackerConfig:
    """Configuration for standalone auto tracker"""

    def __init__(self):
        self.cfg = edict()

        # Model
        self.cfg.MODEL = edict()
        self.cfg.MODEL.BACKBONE = edict()
        self.cfg.MODEL.BACKBONE.TYPE = 'vit_tiny_patch16_224'
        self.cfg.MODEL.BACKBONE.STRIDE = 16
        self.cfg.MODEL.BACKBONE.CE_LOC = []
        self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []

        self.cfg.MODEL.HEAD = edict()
        self.cfg.MODEL.HEAD.TYPE = 'CENTER'
        self.cfg.MODEL.HEAD.NUM_CHANNELS = 256

        self.cfg.MODEL.IS_DISTILL = False
        self.cfg.MODEL.NUM_OBJECT_QUERIES = 1

        # Data configuration (required by build_box_head)
        # 注意：搜索尺寸256对应pos_embed_x=256(16x16)，模板128对应pos_embed_z=64(8x8)
        self.cfg.DATA = edict()
        self.cfg.DATA.SEARCH = edict()
        self.cfg.DATA.SEARCH.SIZE = 256
        self.cfg.DATA.TEMPLATE = edict()
        self.cfg.DATA.TEMPLATE.SIZE = 128

        # Test parameters
        self.cfg.TEST = edict()
        self.cfg.TEST.TEMPLATE_FACTOR = 2.0
        self.cfg.TEST.TEMPLATE_SIZE = 128
        self.cfg.TEST.SEARCH_FACTOR = 5.0
        self.cfg.TEST.SEARCH_SIZE = 256

        # Tracker parameters
        self.template_factor = 2.0
        self.template_size = 128
        self.search_factor = 5.0
        self.search_size = 256
        self.debug = False
        self.save_all_boxes = False


def get_default_config():
    """Get default configuration"""
    return AutoTrackerConfig()
