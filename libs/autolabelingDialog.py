try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    from PyQt4.QtCore import QSettings

import os
from autolabeling import get_model_manager

BB = QDialogButtonBox


class AutoLabelingDialog(QDialog):

    def __init__(self, parent=None, default_save_dir=None, current_folder=None):
        super(AutoLabelingDialog, self).__init__(parent)
        self.setWindowTitle('Auto-labeling / Tracking Configuration')
        self.parent_widget = parent
        self.current_folder = current_folder

        self.available_models = []
        self.all_class_names = set()

        self.save_dir = default_save_dir
        self.selected_roi = None
        self.selected_annotation = None

        self.settings = QSettings('labelImg', 'labelImg')

        if self.save_dir:
            self.load_class_names_from_folder(self.save_dir)

        try:
            manager = get_model_manager()
            self.available_models = manager.get_model_list()
        except Exception as e:
            QMessageBox.critical(self, u'Error', u'Failed to load model config: {}'.format(str(e)))

        self.create_widgets()

    def create_widgets(self):
        layout = QVBoxLayout()

        mode_label = QLabel(u'Mode:')
        self.mode_group = QButtonGroup()
        self.detection_radio = QRadioButton(u'Detection (Object Detection)')
        self.tracking_radio = QRadioButton(u'Tracking (Single Object Tracking)')
        self.detection_radio.setChecked(True)
        self.mode_group.addButton(self.detection_radio, 0)
        self.mode_group.addButton(self.tracking_radio, 1)
        self.detection_radio.toggled.connect(self.on_mode_changed)

        mode_layout = QVBoxLayout()
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.detection_radio)
        mode_layout.addWidget(self.tracking_radio)
        layout.addLayout(mode_layout)

        layout.addWidget(self.create_separator())

        model_label = QLabel(u'Detection Model:')
        self.model_combo = QComboBox()
        for model in self.available_models:
            display_text = u'{} - {}'.format(model['name'], model['description'])
            self.model_combo.addItem(display_text, userData=model['id'])

        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        range_label = QLabel(u'Range:')
        self.range_group = QButtonGroup()
        self.current_image_radio = QRadioButton(u'Current Image')
        self.entire_dataset_radio = QRadioButton(u'Entire Dataset')
        self.current_image_radio.setChecked(True)
        self.range_group.addButton(self.current_image_radio, 0)
        self.range_group.addButton(self.entire_dataset_radio, 1)

        range_layout = QVBoxLayout()
        range_layout.addWidget(range_label)
        range_layout.addWidget(self.current_image_radio)
        range_layout.addWidget(self.entire_dataset_radio)
        layout.addLayout(range_layout)

        self.tracking_group_box = QGroupBox(u'Tracking Configuration')
        tracking_layout = QVBoxLayout()

        iou_label = QLabel(u'IOU Threshold:')
        iou_layout = QHBoxLayout()
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setMinimum(0.0)
        self.iou_spinbox.setMaximum(1.0)
        self.iou_spinbox.setSingleStep(0.05)
        last_iou = self.settings.value('tracking/iou_threshold', 0.8, type=float)
        self.iou_spinbox.setValue(last_iou)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_spinbox)
        iou_layout.addStretch()
        tracking_layout.addLayout(iou_layout)

        conflict_label = QLabel(u'On IOU Conflict:')
        self.conflict_group = QButtonGroup()
        self.keep_existing_radio = QRadioButton(u'Keep Existing Annotations')
        self.use_tracking_radio = QRadioButton(u'Use Tracking Results')
        self.conflict_group.addButton(self.keep_existing_radio, 0)
        self.conflict_group.addButton(self.use_tracking_radio, 1)
        last_conflict = self.settings.value('tracking/conflict_strategy', 'keep_existing')
        if last_conflict == 'use_tracking':
            self.use_tracking_radio.setChecked(True)
        else:
            self.keep_existing_radio.setChecked(True)

        conflict_layout = QVBoxLayout()
        conflict_layout.addWidget(conflict_label)
        conflict_layout.addWidget(self.keep_existing_radio)
        conflict_layout.addWidget(self.use_tracking_radio)
        tracking_layout.addLayout(conflict_layout)

        self.tracking_group_box.setLayout(tracking_layout)
        self.tracking_group_box.setVisible(False)
        layout.addWidget(self.tracking_group_box)

        layout.addWidget(self.create_separator())

        save_path_label = QLabel(u'Save Path:')
        self.save_path_display = QLineEdit()
        self.save_path_display.setReadOnly(True)
        if self.save_dir:
            self.save_path_display.setText(self.save_dir)
        else:
            self.save_path_display.setText(u'(Same as current folder)')
        self.browse_button = QPushButton(u'Browse...')
        self.browse_button.clicked.connect(self.browse_save_path)

        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(save_path_label)
        save_path_layout.addWidget(self.save_path_display)
        save_path_layout.addWidget(self.browse_button)
        layout.addLayout(save_path_layout)

        self.button_box = BB(BB.Ok | BB.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addStretch()
        layout.addWidget(self.button_box)

        self.setLayout(layout)
        self.resize(600, 700)

    def load_class_names_from_folder(self, folder_path):
        """从文件夹的classes.txt中读取类别"""
        classes_txt_path = os.path.join(folder_path, 'classes.txt')
        if os.path.exists(classes_txt_path):
            try:
                with open(classes_txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        class_name = line.strip()
                        if class_name:
                            self.all_class_names.add(class_name)
                print(u'Loaded {} classes from {}'.format(len(self.all_class_names), classes_txt_path))
            except Exception as e:
                print(u'Failed to load classes from {}: {}'.format(classes_txt_path, str(e)))

    def update_class_combo(self):
        """更新类别下拉列表"""
        if not hasattr(self, 'class_combo'):
            return
        self.class_combo.clear()
        if self.all_class_names:
            for class_name in sorted(self.all_class_names):
                self.class_combo.addItem(class_name)
        else:
            self.class_combo.addItem(u'object')

    def create_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def on_mode_changed(self):
        is_tracking = self.tracking_radio.isChecked()
        self.tracking_group_box.setVisible(is_tracking)
        self.model_combo.setEnabled(not is_tracking)
        self.current_image_radio.setEnabled(not is_tracking)
        self.entire_dataset_radio.setEnabled(not is_tracking)
        self.browse_button.setEnabled(True)
        self.save_path_display.setEnabled(False)
        if is_tracking:
            self.entire_dataset_radio.setChecked(True)
        else:
            self.current_image_radio.setChecked(True)


    def browse_save_path(self):
        save_dir = QFileDialog.getExistingDirectory(
            self,
            u'Select Save Directory',
            self.save_dir if self.save_dir else ''
        )
        if save_dir:
            self.save_dir = save_dir
            self.save_path_display.setText(save_dir)
            self.all_class_names.clear()
            self.load_class_names_from_folder(save_dir)
            self.update_class_combo()

    def pop_up(self):
        if self.exec_():
            save_dir = self.save_dir if self.save_dir else self.current_folder
            if not save_dir:
                QMessageBox.warning(self, u'Warning', u'Please select a save directory')
                return None

            if self.tracking_radio.isChecked():
                iou_threshold = self.iou_spinbox.value()
                conflict_strategy = 'use_tracking' if self.use_tracking_radio.isChecked() else 'keep_existing'
                self.settings.setValue('tracking/iou_threshold', iou_threshold)
                self.settings.setValue('tracking/conflict_strategy', conflict_strategy)
                return {
                    'mode': 'tracking',
                    'roi_source': 'manual',
                    'selected_annotation': None,
                    'iou_threshold': iou_threshold,
                    'conflict_strategy': conflict_strategy,
                    'save_dir': save_dir
                }
            else:
                return {
                    'mode': 'detection',
                    'model_id': self.get_selected_model(),
                    'range': self.get_selected_range(),
                    'save_dir': save_dir
                }
        return None

    def get_selected_model(self):
        return self.model_combo.currentData()

    def get_selected_range(self):
        if self.current_image_radio.isChecked():
            return 'current'
        else:
            return 'all'

    def get_selected_save_dir(self):
        return self.save_dir
