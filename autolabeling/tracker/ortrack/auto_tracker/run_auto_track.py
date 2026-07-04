"""
Run automatic tracking on image sequence
使用方法：
    python run_auto_track.py

此模块是完全独立的，可以直接复制到其他项目使用
"""
import os
import sys
from pathlib import Path
import cv2

# 使用内部lib - 无需外部依赖
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from tracker_core import AutoTracker
from config import AutoTrackerConfig
from yolo_exporter import YOLOExporter


def run_tracking(image_dir, init_bbox, output_dir, checkpoint_path, image_pattern='*.jpg'):
    """
    Run automatic tracking on image sequence

    Args:
        image_dir: Directory containing image sequence
        init_bbox: Initial bounding box [x, y, w, h]
        output_dir: Output directory for annotations
        checkpoint_path: Path to checkpoint file
        image_pattern: File pattern for images (default: '*.jpg')
    """
    # Validate inputs
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    # Get list of images
    image_files = sorted(image_dir.glob(image_pattern))
    if not image_files:
        raise ValueError(f'No images found matching pattern {image_pattern} in {image_dir}')

    print(f'[AutoTrack] Found {len(image_files)} images')
    print(f'[AutoTrack] Output directory: {output_dir}')

    # Initialize tracker (use_cuda=False for CPU-only mode)
    config = AutoTrackerConfig()
    tracker = AutoTracker(checkpoint_path, cfg=config, use_cuda=False)

    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        raise ValueError(f'Cannot read image: {image_files[0]}')

    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    H, W, _ = first_image.shape

    # Initialize exporter
    exporter = YOLOExporter(output_dir, class_id=0, image_width=W, image_height=H)

    # Initialize tracker with first frame
    print(f'[AutoTrack] Initializing tracker with bbox: {init_bbox}')
    tracker.initialize(first_image, init_bbox)

    # Export first frame
    exporter.export_bbox(1, init_bbox, filename=f'{image_files[0].stem}.txt')
    print(f'[AutoTrack] Frame 1/{len(image_files)}: bbox={[round(v, 2) for v in init_bbox]}')

    # Track on remaining frames
    for frame_id, image_path in enumerate(image_files[1:], start=2):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'[WARNING] Cannot read image: {image_path}')
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Track
        bbox = tracker.track(image)

        # Export
        exporter.export_bbox(frame_id, bbox, filename=f'{image_path.stem}.txt')

        if frame_id % 10 == 0 or frame_id == len(image_files):
            print(f'[AutoTrack] Frame {frame_id}/{len(image_files)}: bbox={[round(v, 2) for v in bbox]}')

    print(f'[AutoTrack] Tracking complete! Results saved to: {output_dir}')


if __name__ == '__main__':
    # ==================== CONFIGURE HERE ====================
    # 配置参数
    IMAGE_DIR = r'D:\cv\FPV\dataset0\17-2\images'  # 图片序列目录
    CHECKPOINT_PATH = r'D:\cv\labelImg-uav\ORTrack_ep0300.pth.tar'     # 检查点文件路径
    OUTPUT_DIR = r'D:\cv\FPV\dataset0\try'  # 输出标注目录
    INIT_BBOX = [354, 559.5, 40.0, 18.0]                  # 初始框 [x, y, w, h]
    IMAGE_PATTERN = '*.jpg'                          # 图片文件模式

    # ========================================================

    try:
        run_tracking(IMAGE_DIR, INIT_BBOX, OUTPUT_DIR, CHECKPOINT_PATH, IMAGE_PATTERN)
    except Exception as e:
        print(f'[ERROR] {e}')
        import traceback
        traceback.print_exc()
