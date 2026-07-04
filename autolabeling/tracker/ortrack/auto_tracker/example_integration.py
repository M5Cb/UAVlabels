"""
Example: Integrate AutoTracker into custom annotation tool
示例：将AutoTracker集成到自定义标注工具中

此模块演示如何在自定义标注工具中使用AutoTracker
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# 使用内部lib - 无需外部依赖
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from tracker_core import AutoTracker
from config import AutoTrackerConfig
from yolo_exporter import YOLOExporter


class CustomAnnotationTool:
    """
    Custom annotation tool using AutoTracker
    自定义标注工具示例
    """

    def __init__(self, checkpoint_path):
        """
        Initialize annotation tool

        Args:
            checkpoint_path: Path to ORTrack checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.tracker = None
        self.exporter = None
        self.image_dir = None
        self.output_dir = None

    def setup_tracking(self, image_dir, output_dir, image_width, image_height, class_id=0):
        """
        Setup tracking environment

        Args:
            image_dir: Directory containing images
            output_dir: Output directory for annotations
            image_width: Image width
            image_height: Image height
            class_id: Class ID for objects
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)

        # Initialize tracker (use_cuda=False for CPU-only mode)
        config = AutoTrackerConfig()
        self.tracker = AutoTracker(self.checkpoint_path, cfg=config, use_cuda=False)

        # Initialize exporter
        self.exporter = YOLOExporter(
            str(self.output_dir),
            class_id=class_id,
            image_width=image_width,
            image_height=image_height
        )

    def annotate_sequence(self, image_pattern='*.jpg'):
        """
        Annotate entire image sequence

        Args:
            image_pattern: File pattern for images

        Returns:
            results: Dictionary with tracking results
        """
        if self.tracker is None or self.exporter is None:
            raise RuntimeError('Tracker not setup. Call setup_tracking first.')

        image_files = sorted(self.image_dir.glob(image_pattern))
        if not image_files:
            raise ValueError(f'No images found in {self.image_dir}')

        results = {
            'total_frames': len(image_files),
            'processed_frames': 0,
            'bboxes': []
        }

        return results

    def annotate_with_init_box(self, init_bbox, image_pattern='*.jpg', callback=None):
        """
        Annotate sequence with initial bounding box

        Args:
            init_bbox: Initial bbox [x, y, w, h]
            image_pattern: File pattern for images
            callback: Optional callback function for progress
                     called as callback(frame_id, bbox, status)

        Returns:
            results: Dictionary with tracking results
        """
        if self.tracker is None or self.exporter is None:
            raise RuntimeError('Tracker not setup. Call setup_tracking first.')

        image_files = sorted(self.image_dir.glob(image_pattern))
        if not image_files:
            raise ValueError(f'No images found in {self.image_dir}')

        results = {
            'total_frames': len(image_files),
            'processed_frames': 0,
            'bboxes': [],
            'failed_frames': []
        }

        # Initialize with first frame
        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            raise ValueError(f'Cannot read image: {image_files[0]}')

        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

        print(f'[CustomAnnotationTool] Initializing with bbox: {init_bbox}')
        self.tracker.initialize(first_image, init_bbox)
        self.exporter.export_bbox(1, init_bbox, filename=f'{image_files[0].stem}.txt')

        results['bboxes'].append({'frame': 1, 'bbox': init_bbox})
        results['processed_frames'] = 1

        if callback:
            callback(1, init_bbox, 'initialized')

        # Track on remaining frames
        for frame_id, image_path in enumerate(image_files[1:], start=2):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    results['failed_frames'].append(frame_id)
                    if callback:
                        callback(frame_id, None, 'read_error')
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Track
                bbox = self.tracker.track(image)

                # Export
                self.exporter.export_bbox(frame_id, bbox, filename=f'{image_path.stem}.txt')

                results['bboxes'].append({'frame': frame_id, 'bbox': bbox})
                results['processed_frames'] += 1

                if callback:
                    callback(frame_id, bbox, 'success')

            except Exception as e:
                results['failed_frames'].append(frame_id)
                print(f'[WARNING] Frame {frame_id} processing error: {e}')
                if callback:
                    callback(frame_id, None, f'error: {e}')

        return results


def example_usage():
    """
    Example usage of CustomAnnotationTool
    """
    # Configuration
    CHECKPOINT_PATH = r'ORTrack_ep0300.pth.tar'
    IMAGE_DIR = r'D:\cv\labelImg-uav\test_images'
    OUTPUT_DIR = r'D:\cv\labelImg-uav\output_labels'
    INIT_BBOX = [100, 100, 50, 50]  # [x, y, w, h]
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080

    # Create tool
    tool = CustomAnnotationTool(CHECKPOINT_PATH)

    # Setup tracking
    print('[Example] Setting up tracking environment...')
    tool.setup_tracking(IMAGE_DIR, OUTPUT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, class_id=0)

    # Callback for progress
    def progress_callback(frame_id, bbox, status):
        if status == 'success':
            bbox_str = f'[{", ".join(f"{v:.1f}" for v in bbox)}]'
            print(f'[Example] Frame {frame_id:04d}: {status} - bbox={bbox_str}')
        else:
            print(f'[Example] Frame {frame_id:04d}: {status}')

    # Run annotation
    print('[Example] Starting annotation...')
    results = tool.annotate_with_init_box(INIT_BBOX, image_pattern='*.jpg', callback=progress_callback)

    # Print results
    print('\n[Example] Annotation completed!')
    print(f'  Total frames: {results["total_frames"]}')
    print(f'  Processed frames: {results["processed_frames"]}')
    print(f'  Failed frames: {len(results["failed_frames"])}')
    if results['failed_frames']:
        print(f'  Failed frame IDs: {results["failed_frames"]}')
    print(f'  Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    try:
        example_usage()
    except Exception as e:
        print(f'[ERROR] {e}')
        import traceback
        traceback.print_exc()
