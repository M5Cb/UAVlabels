from math import sqrt
from libs.ustr import ustr
import hashlib
import re
import sys

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    QT5 = True
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    QT5 = False

import cv2


def new_icon(icon):
    return QIcon(':/' + icon)


def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class Struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def format_shortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def generate_color_by_text(text):
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, 100)


def have_qstring():
    """p3/qt5 get rid of QString wrapper as py3 has native unicode str type"""
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)


# QT4 has a trimmed method, in QT5 this is called strip
if QT5:
    def trimmed(text):
        return text.strip()
else:
    def trimmed(text):
        return text.trimmed()


def cv2_to_qpixmap(image_cv, max_width=900, max_height=600):
    """将OpenCV图像转换为Qt Pixmap，自动缩放到指定大小"""
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = cv2.resize(image_rgb, (new_w, new_h))

    q_img = QImage(image_resized.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


def draw_bbox_on_image(image_cv, bbox, class_label, color=(0, 255, 0)):
    """在图像上绘制跟踪框、四角标记和标签"""
    if not bbox:
        return image_cv

    x, y, w, h = [int(v) for v in bbox]

    cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, 3)

    corner_size = 10
    cv2.line(image_cv, (x, y), (x + corner_size, y), color, 2)
    cv2.line(image_cv, (x, y), (x, y + corner_size), color, 2)
    cv2.line(image_cv, (x + w, y), (x + w - corner_size, y), color, 2)
    cv2.line(image_cv, (x + w, y), (x + w, y + corner_size), color, 2)
    cv2.line(image_cv, (x, y + h), (x + corner_size, y + h), color, 2)
    cv2.line(image_cv, (x, y + h), (x, y + h - corner_size), color, 2)
    cv2.line(image_cv, (x + w, y + h), (x + w - corner_size, y + h), color, 2)
    cv2.line(image_cv, (x + w, y + h), (x + w, y + h - corner_size), color, 2)

    label_text = class_label if class_label else "object"
    label_y = max(y - 15, 20)

    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(image_cv, (x, label_y - text_size[1] - 5),
                            (x + text_size[0] + 5, label_y + 5), color, -1)
    cv2.putText(image_cv, label_text, (x + 3, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image_cv
