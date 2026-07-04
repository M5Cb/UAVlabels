# -*- coding: utf-8 -*-

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import os


TRACKER_CONFIG_KEY = 'tracker_type'
CUDA_CONFIG_KEY = 'tracker_use_cuda'


class TrackerSelectDialog(QDialog):
    """追踪器选择对话框 - 让用户选择CSRT或ORTrack"""

    def __init__(self, parent=None, settings=None):
        super(TrackerSelectDialog, self).__init__(parent)
        self.setWindowTitle('Select Tracker')
        self.setModal(True)
        self.setMinimumWidth(550)

        self.settings = settings

        # 从设置中加载上次的选择，如果没有则使用默认值
        self.selected_tracker = 'csrt'
        self.use_cuda = True

        if self.settings:
            self.selected_tracker = self.settings.get(TRACKER_CONFIG_KEY, 'csrt')
            self.use_cuda = self.settings.get(CUDA_CONFIG_KEY, True)

        self.create_widgets()
        self.load_previous_selection()

    def create_widgets(self):
        layout = QVBoxLayout()

        # 说明文本
        info_label = QLabel(
            'Choose a tracker for automatic annotation:\n\n'
            '• CSRT: Fast and lightweight, works on CPU\n'
            '• ORTrack: High-precision with Vision Transformer'
        )
        info_label.setStyleSheet("color: #333; font-size: 11px;")
        layout.addWidget(info_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 创建按钮组以确保互斥
        self.tracker_button_group = QButtonGroup()

        # CSRT选项
        csrt_group = self.create_tracker_group(
            'CSRT (Recommended for quick annotation)',
            'Fast correlation filter based tracker',
            'csrt',
            require_model=False
        )
        layout.addWidget(csrt_group)

        layout.addSpacing(10)

        # ORTrack选项
        ortrack_group = self.create_tracker_group(
            'ORTrack (High precision)',
            'Vision Transformer based tracker',
            'ortrack',
            require_model=False
        )
        layout.addWidget(ortrack_group)

        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)

        # GPU选项（仅for ORTrack）
        gpu_layout = QHBoxLayout()
        self.gpu_checkbox = QCheckBox('Use GPU (CUDA) for ORTrack')
        self.gpu_checkbox.setChecked(self.use_cuda)
        gpu_layout.addWidget(self.gpu_checkbox)
        gpu_layout.addStretch()
        layout.addLayout(gpu_layout)

        layout.addSpacing(10)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton('OK')
        cancel_button = QPushButton('Cancel')

        ok_button.clicked.connect(self.on_ok)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_tracker_group(self, title, description, tracker_type, require_model=False):
        """创建追踪器选项组"""
        group = QGroupBox(title)
        group_layout = QVBoxLayout()

        # 选择按钮
        radio_button = QRadioButton(description)
        radio_button.toggled.connect(lambda checked: self.on_tracker_selected(tracker_type, checked))
        group_layout.addWidget(radio_button)

        # 添加到按钮组以确保互斥
        button_id = 0 if tracker_type == 'csrt' else 1
        self.tracker_button_group.addButton(radio_button, button_id)

        # 存储radio按钮引用
        if tracker_type == 'csrt':
            self.csrt_radio = radio_button
        else:
            self.ortrack_radio = radio_button

        group.setLayout(group_layout)
        return group

    def on_tracker_selected(self, tracker_type, checked):
        """当选择追踪器时的回调"""
        if checked:
            self.selected_tracker = tracker_type

    def load_previous_selection(self):
        """从之前的选择中恢复UI状态"""
        if self.selected_tracker == 'csrt':
            self.csrt_radio.setChecked(True)
        else:
            self.ortrack_radio.setChecked(True)

    def on_ok(self):
        """保存选择并关闭对话框"""
        if self.settings:
            self.settings[TRACKER_CONFIG_KEY] = self.selected_tracker
            self.settings[CUDA_CONFIG_KEY] = self.gpu_checkbox.isChecked()
            self.settings.save()
        self.accept()


    def get_tracker_config(self):
        """获取选择的追踪器配置"""
        config = {
            'tracker_type': self.selected_tracker,
            'use_cuda': self.gpu_checkbox.isChecked()
        }

        if self.selected_tracker == 'ortrack':
            config['checkpoint_path'] = 'autolabeling\\models\\ORTrack_ep0300.pth.tar'

        return config
