from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from LabelItem import ImageItem

#展示示图


ZOOMMAX = 50
ZOOMMIN = 0.02

class MyView(QGraphicsView, QObject):

    sendNext = pyqtSignal(int)
    sendUp = pyqtSignal(int)

    def __init__(self):
        super(MyView, self).__init__()
        self.initUI()

    def initUI(self):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.ZoomValue = 1.0
        self.name = None
        self.secne = QGraphicsScene(-100000, -100000, 200000, 200000, self)
        self.widget = QWidget(self)
        self.PosInfoWidget = QWidget(self)
        self.PosInfoLabel = QLabel(self)
        self.setScene(self.secne)
        self.imageItem = ImageItem.LabelablePixmapItem()
        self.secne.addItem(self.imageItem)



        self.tilePixmap = QPixmap(36, 36)
        self.PosInfoLabel.setText("W:0,H:0 | X:0,Y:0 | R:0,G:0,B:0")
        self.PosInfoLabel.setStyleSheet("color:rgb(200,255,200); "
                                        "background-color:rgba(50,50,50,160); "
                                        "font: Microsoft YaHei;"
                                        "font-size: 15px;")
        self.PosInfoWidget.setFixedHeight(25)
        self.setBackground(True, False)
        self.PosInfoWidget.setGeometry(0, self.height() - 25, self.width(), 25)
        self.PosInfoWidget.setStyleSheet("background-color:rgba(0,0,0,0);")
        self.InfoLayout = QHBoxLayout()
        self.InfoLayout.setSpacing(0)
        self.InfoLayout.setContentsMargins(0, 0, 0, 0)
        self.InfoLayout.addWidget(self.PosInfoLabel)
        self.PosInfoWidget.setLayout(self.InfoLayout)
        # self.imageItem.signal.sendInfo.connect(lambda Info: self.PosInfoLabel.setText(Info))

    def setProject(self, name):
        self.name = name

    def setBackground(self, onBtn, inv):
        if onBtn:
            tilePainter = QPainter(self.tilePixmap)
            fill_color = QColor(220, 220, 220) if inv else QColor(35, 35, 35)
            self.tilePixmap.fill(fill_color)
            color = QColor(50, 50, 50, 255)
            invertedColor = QColor(210, 210, 210, 255)
            tilePainter.fillRect(0, 0, 18, 18,  invertedColor if inv else color)
            tilePainter.fillRect(18, 18, 18, 18, invertedColor if inv else color)
            tilePainter.end()

    def paintEvent(self, event):
        painter = QPainter(self.viewport())
        painter.drawTiledPixmap(QRect(QPoint(0, 0), QPoint(self.width(), self.height())), self.tilePixmap)
        super(MyView, self).paintEvent(event)

    def setImage(self, path):
        self.imageItem.setPixmap(QPixmap(path))

        # self.path = path
        self.onCenter()

    def onCenter(self):
        self.fitFraame()
        self.centerOn(self.imageItem.pixmap().width()/2, self.imageItem.pixmap().height()/2)
        self.imageItem.setPos(0, 0)

    def resizeEvent(self, event):
        self.fitFraame()
        self.onCenter()
        self.PosInfoWidget.setGeometry(0, self.height() - 25, self.width(), 25)
        super(MyView, self).resizeEvent(event)


    def fitFraame(self):
        pix = self.imageItem.pixmap()
        winWidth = self.width()
        winHeight = self.height()
        ScaleWidth = (pix.width() + 1) / winWidth
        ScaleHeight = (pix.height() + 1) / winHeight
        s_temp = 1 / ScaleWidth if ScaleWidth >= ScaleHeight else 1 / ScaleHeight
        scale = s_temp / self.ZoomValue
        if scale >= ZOOMMAX or scale <= ZOOMMIN:
            return
        self.onZoom(scale)
        self.ZoomValue = s_temp

    def onZoom(self, scaleFactor):
        self.ZoomValue *= scaleFactor
        self.scale(scaleFactor, scaleFactor)

    def wheelEvent(self, event):
        scrollAmount = event.angleDelta()
        if scrollAmount.y() > 0 and self.ZoomValue >= ZOOMMAX:
            return
        if scrollAmount.y() < 0 and self.ZoomValue <= ZOOMMIN:
            return
        if scrollAmount.y() > 0:
            self.onZoom(1.1)
        else:
            self.onZoom(0.9)


    def set_pixmap_item(self, item):
        self.secne.clear()
        self.imageItem = item
        self.secne.addItem(item)
        self.onCenter()

    def set_labelable_pixmap(self, pixmap_item):
        """设置可标注的图片项"""
        self.scene.addItem(pixmap_item)

    def setMode(self, mode):
        self.imageItem.set_annotation_mode(mode)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.sendUp.emit(-1)
            event.accept()
        elif event.key() == Qt.Key_Down:
            self.sendNext.emit(1)
            event.accept()
        elif event.key() == Qt.Key_Escape and self.imageItem.cancel_current_annotation():
            event.accept()
        else:
            super().keyPressEvent(event)

