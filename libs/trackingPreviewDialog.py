# -*- coding: utf-8 -*-

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import cv2
import os
from libs.utils import cv2_to_qpixmap, draw_bbox_on_image


class TrackingPreviewDialog(QDialog):
    """跟踪过程实时预览对话框"""

    def __init__(self, parent=None):
        super(TrackingPreviewDialog, self).__init__(parent)
        self.setWindowTitle('Tracking in Progress')
        self.stopped = False
        self.current_frame_idx = 0
        self.total_frames = 0

        # 模式管理属性
        self.mode = 'tracking'  # 'tracking' 或 'viewing'
        self.tracking_results = []  # 存储所有帧的跟踪结果
        self.view_frame_idx = 0  # 当前查看的帧索引
        self.truncate_frame_idx = None  # 用户标记的截断帧

        self.resize(1000, 750)
        self.create_widgets()

    def create_widgets(self):
        layout = QVBoxLayout()

        info_layout = QHBoxLayout()
        self.info_label = QLabel()
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(900, 600)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        layout.addWidget(self.image_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.stop_button = QPushButton('⏹ Stop Tracking')
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_button.clicked.connect(self.stop_tracking)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_frame(self, frame_path, current, total, bbox=None, success=True):
        """
        更新显示的帧

        参数:
            frame_path: 当前帧的文件路径
            current: 当前帧索引
            total: 总帧数
            bbox: 跟踪框 (x, y, w, h) 或 None
            success: 是否跟踪成功
        """
        self.current_frame_idx = current
        self.total_frames = total

        progress = int((current * 100) / total) if total > 0 else 0
        self.progress_bar.setValue(progress)

        frame_name = os.path.basename(frame_path)
        self.info_label.setText(
            f"Frame: {frame_name} ({current}/{total})"
        )

        image_cv = cv2.imread(frame_path)
        if image_cv is not None:
            image_cv = draw_bbox_on_image(image_cv, bbox, "Tracking", (0, 255, 0))
            pixmap = cv2_to_qpixmap(image_cv, max_width=900, max_height=600)
            self.image_label.setPixmap(pixmap)

        # 保存到tracking_results (用于查看模式)
        if self.mode == 'tracking':
            frame_result = {
                'frame_path': frame_path,
                'bbox': bbox
            }
            if current - 1 < len(self.tracking_results):
                self.tracking_results[current - 1] = frame_result
            else:
                self.tracking_results.append(frame_result)

        QApplication.processEvents()

    def stop_tracking(self):
        """停止跟踪"""
        self.stopped = True
        self.stop_button.setEnabled(False)
        self.stop_button.setText('⏹ Stopping...')

    def is_stopped(self):
        """检查是否已停止"""
        return self.stopped

    def get_stopped_frame_index(self):
        """获取停止时的帧索引"""
        return self.current_frame_idx

    def tracking_completed(self, all_tracking_results):
        """跟踪完成，进入查看模式

        参数:
            all_tracking_results: 所有帧的跟踪结果列表
        """
        self.tracking_results = all_tracking_results
        self.mode = 'viewing'
        self.stopped = True

        # 隐藏Stop按钮
        self.stop_button.hide()

        # 初始化查看模式（从第二帧开始，跳过初始化帧）
        self.view_frame_idx = 1 if len(all_tracking_results) > 1 else 0
        self.truncate_frame_idx = None

        # 显示第二帧（或第一帧，如果只有一帧）
        if all_tracking_results:
            self.display_frame(self.view_frame_idx)

        # 更新窗口标题
        self.setWindowTitle('Tracking Results')

    def keyPressEvent(self, event):
        """处理快捷键输入"""
        if self.mode == 'viewing':
            if event.key() == Qt.Key_A:
                self.view_prev_frame()
            elif event.key() == Qt.Key_D:
                self.view_next_frame()
            elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                if self.view_frame_idx == 0:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(self, u'Info', u'Cannot truncate at initialization frame, please select a frame after the first one')
                else:
                    self.confirm_truncate()
            elif event.key() == Qt.Key_Escape:
                self.reject()
            else:
                super(TrackingPreviewDialog, self).keyPressEvent(event)
        else:
            super(TrackingPreviewDialog, self).keyPressEvent(event)

    def view_prev_frame(self):
        """显示上一帧"""
        if self.view_frame_idx > 0:
            self.view_frame_idx -= 1
            self.display_frame(self.view_frame_idx)

    def view_next_frame(self):
        """显示下一帧"""
        if self.view_frame_idx < len(self.tracking_results) - 1:
            self.view_frame_idx += 1
            self.display_frame(self.view_frame_idx)

    def display_frame(self, idx):
        """显示指定帧的跟踪结果

        参数:
            idx: 帧索引
        """
        if idx < 0 or idx >= len(self.tracking_results):
            return

        frame_result = self.tracking_results[idx]
        frame_path = frame_result.get('frame_path', '')
        bbox = frame_result.get('raw_bbox')
        total = len(self.tracking_results)

        # 更新进度条
        progress = int((idx * 100) / total) if total > 0 else 0
        self.progress_bar.setValue(progress)

        # 构建info文本
        frame_name = os.path.basename(frame_path) if frame_path else 'Unknown'

        # 在查看模式下，区分初始化帧和后续帧
        if idx == 0:
            hint_text = u"[初始化帧 - Initialization Frame] | A/D翻帧"
        else:
            hint_text = u"按Enter在此帧截断 | A/D翻帧"
        info_text = u"Frame: {} ({}/{})\n{}".format(
            frame_name, idx + 1, total, hint_text
        )
        self.info_label.setText(info_text)

        # 显示图像和标注框
        if frame_path and os.path.exists(frame_path):
            image_cv = cv2.imread(frame_path)
            if image_cv is not None:
                image_cv = draw_bbox_on_image(image_cv, bbox, "Tracking", (0, 255, 0))
                pixmap = cv2_to_qpixmap(image_cv, max_width=900, max_height=600)
                self.image_label.setPixmap(pixmap)

        QApplication.processEvents()

    def confirm_truncate(self):
        """确认截断点，关闭对话框

        记录用户选择的截断帧，后续labelImg.py会读取此值
        """
        self.truncate_frame_idx = self.view_frame_idx
        self.accept()
