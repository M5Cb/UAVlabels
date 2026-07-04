"""
YOLO format exporter for AutoTracker
"""
import os
from pathlib import Path


class YOLOExporter:
    """
    Export tracking results in YOLO format
    YOLO format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    """

    def __init__(self, output_dir, class_id=0, image_width=None, image_height=None):
        """
        Initialize YOLO exporter

        Args:
            output_dir: Output directory for annotation files
            class_id: Class ID for all objects (default 0)
            image_width: Image width (optional, needed for normalization)
            image_height: Image height (optional, needed for normalization)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_id = class_id
        self.image_width = image_width
        self.image_height = image_height

    def export_bbox(self, frame_id, bbox, image_width=None, image_height=None, filename=None):
        """
        Export single bounding box in YOLO format

        Args:
            frame_id: Frame number/ID
            bbox: Bounding box [x, y, w, h] in pixel coordinates
            image_width: Image width for normalization (uses self.image_width if None)
            image_height: Image height for normalization (uses self.image_height if None)
            filename: Custom filename (uses frame_id if None)

        Returns:
            annotation_file: Path to saved annotation file
        """
        if image_width is None:
            image_width = self.image_width
        if image_height is None:
            image_height = self.image_height

        if image_width is None or image_height is None:
            raise ValueError('Image dimensions must be provided for normalization')

        # Convert bbox from [x, y, w, h] to YOLO format
        x, y, w, h = bbox
        x_center = (x + w / 2.0) / image_width
        y_center = (y + h / 2.0) / image_height
        width_norm = w / image_width
        height_norm = h / image_height

        # Clamp normalized values to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        # Generate filename
        if filename is None:
            filename = f'{frame_id:06d}.txt'
        elif not filename.endswith('.txt'):
            filename = f'{filename}.txt'

        # Write annotation file
        annotation_file = self.output_dir / filename
        with open(annotation_file, 'w') as f:
            f.write(f'{self.class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n')

        return annotation_file

    def export_multiple_bboxes(self, frame_id, bboxes, image_width=None, image_height=None, filename=None):
        """
        Export multiple bounding boxes in YOLO format

        Args:
            frame_id: Frame number/ID
            bboxes: List of bounding boxes [[x, y, w, h], ...]
            image_width: Image width for normalization
            image_height: Image height for normalization
            filename: Custom filename

        Returns:
            annotation_file: Path to saved annotation file
        """
        if image_width is None:
            image_width = self.image_width
        if image_height is None:
            image_height = self.image_height

        if image_width is None or image_height is None:
            raise ValueError('Image dimensions must be provided for normalization')

        # Generate filename
        if filename is None:
            filename = f'{frame_id:06d}.txt'
        elif not filename.endswith('.txt'):
            filename = f'{filename}.txt'

        annotation_file = self.output_dir / filename
        with open(annotation_file, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                x_center = (x + w / 2.0) / image_width
                y_center = (y + h / 2.0) / image_height
                width_norm = w / image_width
                height_norm = h / image_height

                # Clamp normalized values
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))

                f.write(f'{self.class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n')

        return annotation_file
