"""
Auto Tracker Module - Standalone tracking module for ORTrack
完全独立的自动跟踪模块，不依赖外部lib文件夹
"""
import os
import sys

# Add this directory to path so internal lib can be imported
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from .tracker_core import AutoTracker
from .config import AutoTrackerConfig

__all__ = ['AutoTracker', 'AutoTrackerConfig']
