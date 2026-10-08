from  PyQt5.QtCore import *
from  PyQt5.QtWidgets import *
from  PyQt5.QtGui import *


# 自定义等比例缩放QLabel

class ScaledLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self._scale_pixmap()

    def _scale_pixmap(self):
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self._scale_pixmap()
        super().resizeEvent(event)