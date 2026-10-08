from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject
from PyQt5.QtWidgets import (QGraphicsObject, QMenu, QAction, QGraphicsPixmapItem,
                             QDialog, QVBoxLayout, QLineEdit, QPushButton, QListWidget,
                             QLabel, QMessageBox, QGraphicsItem, QGraphicsLineItem)
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QCursor, QFont, QPolygonF, QPainterPath
import os.path
import json

# --- 优化：HSV黄金分割动态配色（无限类别不撞色） ---
def get_label_color(label, categories):
    try: idx = categories.index(label)
    except: idx = 0
    h = (idx * 0.618033988749895) % 1.0
    color = QColor(); color.setHsvF(h, 0.85, 0.95)
    return color

class PixmapSignalProxy(QObject):
    send_go = pyqtSignal(str)
    annotation_updated = pyqtSignal()
    annotation_deleted = pyqtSignal()

class CategoryDialog(QDialog):
    send_content = pyqtSignal(str)
    def __init__(self, existing_categories=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标注类别管理"); self.setMinimumSize(300, 400)
        self.categories = existing_categories if existing_categories else []
        layout = QVBoxLayout(); self.edit_category = QLineEdit()
        self.edit_category.setPlaceholderText("输入新类别后点击确认")
        self.btn_confirm = QPushButton("确认添加"); self.btn_confirm.clicked.connect(self.add_category)
        self.list_widget = QListWidget(); self.list_widget.itemDoubleClicked.connect(self.accept_selected)
        self.update_list()
        layout.addWidget(QLabel("新类别:")); layout.addWidget(self.edit_category); layout.addWidget(self.btn_confirm)
        layout.addWidget(QLabel("双击选择已有类别:")); layout.addWidget(self.list_widget)
        self.setLayout(layout); self.selected_category = ""
    def add_category(self):
        new = self.edit_category.text().strip()
        if new and new not in self.categories:
            self.send_content.emit(new); self.categories.append(new); self.update_list(); self.edit_category.clear()
    def update_list(self): self.list_widget.clear(); self.list_widget.addItems(self.categories)
    def accept_selected(self, item): self.selected_category = item.text(); self.accept()
    def get_selected_category(self): return self.selected_category


class BaseAnnotation(QGraphicsObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.category = ""
        self.category_font = QFont("微软雅黑", 10, QFont.Bold)
        self.deletable = False
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)

    def activate_interaction(self, enabled=True):
        self.deletable = True
        self.set_interaction_enabled(enabled)

    def set_interaction_enabled(self, enabled):
        """Let drawing clicks pass through completed annotations when disabled."""
        if not self.deletable:
            self.setAcceptedMouseButtons(Qt.NoButton)
            self.setAcceptHoverEvents(False)
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.setFlag(QGraphicsItem.ItemIsSelectable, False)
            return
        if not enabled:
            self.setSelected(False)
            self.unsetCursor()
        buttons = Qt.RightButton
        if enabled:
            buttons |= Qt.LeftButton
        self.setAcceptedMouseButtons(buttons)
        self.setAcceptHoverEvents(enabled)
        self.setFlag(QGraphicsItem.ItemIsMovable, enabled)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def set_category(self, category):
        self.category = category
        self.update()

    def paint(self, painter, option, widget):
        # 这里的绘制逻辑保持不变...
        color = get_label_color(self.category, self.parentItem().existing_categories if self.parentItem() else [])
        pen = QPen(color, 2)
        if self.isSelected():
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        brush_color = QColor(color)
        brush_color.setAlpha(40)
        painter.setBrush(QBrush(brush_color))

        if self.category:
            painter.setPen(QPen(color))
            painter.setFont(self.category_font)
            rect = self.boundingRect().adjusted(5, 30, -5, -5)
            painter.drawText(QPointF(rect.x(), rect.y() - 5), self.category)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.setSelected(True)
            event.accept()
            return
        if event.button() == Qt.LeftButton and (
            self.flags() & QGraphicsItem.ItemIsSelectable
        ):
            self.setSelected(True)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.RightButton and self.deletable:
            self.delete_annotation()
        elif event.button() == Qt.LeftButton:
            self.finalize_annotation()
        event.accept()

    # --- 核心修复函数 ---
    def delete_annotation(self):
        parent = self.parentItem()  # 提前抓取父对象
        if parent and hasattr(parent, 'annotations'):
            if self in parent.annotations:
                parent.annotations.remove(self)

        scene = self.scene()
        if scene:
            scene.removeItem(self)

        # 确保 parent 不为空才调用 signal_proxy
        if parent and hasattr(parent, 'signal_proxy'):
            parent.signal_proxy.annotation_deleted.emit()
            parent.signal_proxy.annotation_updated.emit()

    def finalize_annotation(self):
        parent = self.parentItem()
        if not parent: return

        dialog = CategoryDialog(parent.existing_categories)
        if dialog.exec_() == QDialog.Accepted:
            selected_category = dialog.get_selected_category()
            if selected_category:
                self.set_category(selected_category)
                if selected_category not in parent.existing_categories:
                    parent.existing_categories.append(selected_category)

        if parent and hasattr(parent, 'signal_proxy'):
            parent.signal_proxy.annotation_updated.emit()

class RectAnnotation(BaseAnnotation):
    def __init__(self, start_pos: QPointF, parent=None):
        super().__init__(parent); self.rect = QRectF(start_pos, QPointF(start_pos.x()+1, start_pos.y()+1))
        self.active_handle = 0; self.margin = 10
    def update_rect(self, end_pos: QPointF): self.prepareGeometryChange(); self.rect.setBottomRight(end_pos)
    def boundingRect(self): return self.rect.normalized().adjusted(-15, -40, 15, 15)
    def paint(self, painter, option, widget): super().paint(painter, option, widget); painter.drawRect(self.rect.normalized())
    def _get_handle(self, pos):
        r = self.rect.normalized(); m = self.margin
        if (pos - r.topLeft()).manhattanLength() < m*1.5: return 5
        if (pos - r.topRight()).manhattanLength() < m*1.5: return 6
        if (pos - r.bottomLeft()).manhattanLength() < m*1.5: return 7
        if (pos - r.bottomRight()).manhattanLength() < m*1.5: return 8
        if abs(pos.y() - r.top()) < m: return 1
        if abs(pos.y() - r.bottom()) < m: return 2
        if abs(pos.x() - r.left()) < m: return 3
        if abs(pos.x() - r.right()) < m: return 4
        return 0
    def hoverMoveEvent(self, event):
        if self.isSelected():
            h = self._get_handle(event.pos())
            cursors = {1:Qt.SizeVerCursor, 2:Qt.SizeVerCursor, 3:Qt.SizeHorCursor, 4:Qt.SizeHorCursor, 5:Qt.SizeBDiagCursor, 8:Qt.SizeBDiagCursor, 6:Qt.SizeFDiagCursor, 7:Qt.SizeFDiagCursor}
            self.setCursor(cursors.get(h, Qt.ArrowCursor))
        super().hoverMoveEvent(event)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isSelected():
            self.active_handle = self._get_handle(event.pos())
            if self.active_handle > 0: event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self.active_handle > 0:
            self.prepareGeometryChange(); p = event.pos(); r = self.rect
            if self.active_handle == 1: r.setTop(p.y())
            elif self.active_handle == 2: r.setBottom(p.y())
            elif self.active_handle == 3: r.setLeft(p.x())
            elif self.active_handle == 4: r.setRight(p.x())
            elif self.active_handle == 5: r.setTopLeft(p)
            elif self.active_handle == 6: r.setTopRight(p)
            elif self.active_handle == 7: r.setBottomLeft(p)
            elif self.active_handle == 8: r.setBottomRight(p)
            self.rect = r; return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if self.active_handle > 0: self.active_handle = 0; self.rect = self.rect.normalized(); self.parentItem().signal_proxy.annotation_updated.emit()
        super().mouseReleaseEvent(event)
    def to_dict(self):
        r = self.rect.normalized(); o = self.pos(); img = self.parentItem().pixmap().size()
        return {"type": "rect", "lable": self.category, "points": [{"x": (r.left()+o.x())/img.width(), "y": (r.top()+o.y())/img.height()}, {"x": (r.right()+o.x())/img.width(), "y": (r.bottom()+o.y())/img.height()}]}

class PolygonAnnotation(BaseAnnotation):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.completed = False
        self.temp_point = None
        self.start_point_highlighted = False
        self.start_point_radius = 5.0
        self.setZValue(10)

    def add_point(self, pos: QPointF):
        self.prepareGeometryChange()
        self.points.append(QPointF(pos))

    def set_temp(self, pos):
        self.prepareGeometryChange()
        self.temp_point = QPointF(pos)

    def set_start_point_highlighted(self, highlighted, radius):
        radius = max(float(radius), 1.0)
        if (
            self.start_point_highlighted == highlighted
            and abs(self.start_point_radius - radius) < 0.01
        ):
            return
        self.prepareGeometryChange()
        self.start_point_highlighted = highlighted
        self.start_point_radius = radius
        self.update()

    def is_near_start(self, pos, tolerance):
        if len(self.points) < 3:
            return False
        delta = pos - self.points[0]
        return delta.x() * delta.x() + delta.y() * delta.y() <= tolerance * tolerance

    def complete(self):
        cleaned = []
        for point in self.points:
            if not cleaned or point != cleaned[-1]:
                cleaned.append(QPointF(point))
        if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned.pop()
        if len(cleaned) < 3:
            return False

        area_twice = 0.0
        for index, point in enumerate(cleaned):
            next_point = cleaned[(index + 1) % len(cleaned)]
            area_twice += point.x() * next_point.y() - next_point.x() * point.y()
        if abs(area_twice) < 0.000001:
            return False

        self.prepareGeometryChange()
        self.points = cleaned
        self.completed = True
        self.temp_point = None
        self.start_point_highlighted = False
        self.update()
        return True

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        if self.points:
            path = QPainterPath(); path.moveTo(self.points[0])
            for p in self.points[1:]: path.lineTo(p)
            if self.temp_point and not self.completed: path.lineTo(self.temp_point)
            if self.completed: path.closeSubpath()
            painter.drawPath(path)
            if not self.completed and len(self.points) >= 3:
                color = get_label_color(
                    self.category,
                    self.parentItem().existing_categories if self.parentItem() else []
                )
                radius = self.start_point_radius * (1.35 if self.start_point_highlighted else 1.0)
                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(color if self.start_point_highlighted else Qt.NoBrush))
                painter.drawEllipse(self.points[0], radius, radius)

    def boundingRect(self):
        if not self.points: return QRectF()
        visible_points = list(self.points)
        if self.temp_point and not self.completed:
            visible_points.append(self.temp_point)
        xs = [p.x() for p in visible_points]; ys = [p.y() for p in visible_points]
        margin = max(5.0, self.start_point_radius * 1.5 + 2.0)
        return QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)).adjusted(-margin, -35, margin, margin)
    def to_dict(self):
        img = self.parentItem().pixmap().size(); o = self.pos()
        return {"type": "polygon", "lable": self.category, "points": [{"x": (p.x()+o.x())/img.width(), "y": (p.y()+o.y())/img.height()} for p in self.points]}

class LabelablePixmapItem(QGraphicsPixmapItem):
    class AnnotationMode: NONE = 0; RECTANGLE = 1; POLYGON = 2
    def __init__(self):
        super().__init__(); self.signal_proxy = PixmapSignalProxy(); self.setAcceptHoverEvents(True)
        self.current_mode = 0; self.current_annotation = None; self.annotations = []; self.existing_categories = []
        self.v_line = QGraphicsLineItem(self); self.h_line = QGraphicsLineItem(self)
        lp = QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine)
        self.v_line.setPen(lp); self.h_line.setPen(lp); self.v_line.setZValue(999); self.v_line.hide(); self.h_line.hide()

    def setPath(self, path): self.path = path
    def initCatories(self, cats): self.existing_categories = cats
    def set_annotation_mode(self, mode):
        if mode != self.current_mode:
            self.cancel_current_annotation()
        self.current_mode = mode
        interaction_enabled = mode == self.AnnotationMode.NONE
        for annotation in self.annotations:
            annotation.set_interaction_enabled(interaction_enabled)

    def _view_scale(self):
        if not self.scene() or not self.scene().views():
            return 1.0
        transform = self.scene().views()[0].transform()
        return max(abs(transform.m11()), abs(transform.m22()), 0.000001)

    def _polygon_close_tolerance(self):
        return 10.0 / self._view_scale()

    def _finish_current_polygon(self):
        annotation = self.current_annotation
        if not isinstance(annotation, PolygonAnnotation) or not annotation.complete():
            return False
        annotation.finalize_annotation()
        annotation.activate_interaction(
            self.current_mode == self.AnnotationMode.NONE
        )
        self.annotations.append(annotation)
        self.current_annotation = None
        self.signal_proxy.annotation_updated.emit()
        return True

    def cancel_current_annotation(self):
        annotation = self.current_annotation
        if annotation is None:
            return False
        self.current_annotation = None
        scene = annotation.scene()
        if scene:
            scene.removeItem(annotation)
        return True

    def hoverMoveEvent(self, event):
        p = event.pos()
        if self.current_mode != 0:
            self.v_line.setLine(p.x(), 0, p.x(), self.pixmap().height()); self.h_line.setLine(0, p.y(), self.pixmap().width(), p.y())
            self.v_line.show(); self.h_line.show()
        else: self.v_line.hide(); self.h_line.hide()
        sp = self.mapToScene(p)
        if self.current_mode == 1 and self.current_annotation: self.current_annotation.update_rect(sp)
        elif self.current_mode == 2 and isinstance(self.current_annotation, PolygonAnnotation):
            self.current_annotation.set_temp(sp)
            tolerance = self._polygon_close_tolerance()
            self.current_annotation.set_start_point_highlighted(
                self.current_annotation.is_near_start(sp, tolerance),
                5.0 / self._view_scale()
            )
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        if self.current_mode == 1 and event.button() == Qt.LeftButton:
            if not self.current_annotation:
                self.current_annotation = RectAnnotation(sp, self); self.scene().addItem(self.current_annotation)
            else:
                self.current_annotation.update_rect(sp); self.current_annotation.finalize_annotation()
                self.current_annotation.activate_interaction(
                    self.current_mode == self.AnnotationMode.NONE
                )
                self.annotations.append(self.current_annotation); self.current_annotation = None; self.signal_proxy.annotation_updated.emit()
        elif self.current_mode == 2:
            if event.button() == Qt.LeftButton:
                if not self.current_annotation:
                    self.current_annotation = PolygonAnnotation(self); self.scene().addItem(self.current_annotation)
                elif self.current_annotation.is_near_start(
                    sp, self._polygon_close_tolerance()
                ):
                    self._finish_current_polygon()
                    event.accept()
                    return
                self.current_annotation.add_point(sp)
        else: super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            self.current_mode == self.AnnotationMode.POLYGON
            and event.button() == Qt.LeftButton
            and self.current_annotation
        ):
            self._finish_current_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def remove_annotations(self):
        for a in self.annotations: self.scene().removeItem(a)
        self.annotations.clear(); self.signal_proxy.annotation_updated.emit()
    def save_annotations(self, path):
        img = self.pixmap().size()
        data = {"image_width": img.width(), "image_height": img.height(), "annotations": [a.to_dict() for a in self.annotations]}
        with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
    def load_annotations(self, path):
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            img = self.pixmap().size()
            for a in data["annotations"]:
                if a["type"] == "rect":
                    pts = a["points"]; s = QPointF(pts[0]["x"]*img.width(), pts[0]["y"]*img.height()); e = QPointF(pts[1]["x"]*img.width(), pts[1]["y"]*img.height())
                    category = a.get("lable") or a.get("label") or a.get("category") or ""
                    r = RectAnnotation(s, parent=self); r.rect = QRectF(s, e); r.category = category
                    if category and category not in self.existing_categories:
                        self.existing_categories.append(category)
                    r.activate_interaction(self.current_mode == self.AnnotationMode.NONE)
                    self.annotations.append(r); self.scene().addItem(r)
                elif a["type"] == "polygon":
                    poly = PolygonAnnotation(parent=self)
                    for p in a["points"]: poly.add_point(QPointF(p["x"]*img.width(), p["y"]*img.height()))
                    category = a.get("lable") or a.get("label") or a.get("category") or ""
                    poly.category = category; poly.complete()
                    if category and category not in self.existing_categories:
                        self.existing_categories.append(category)
                    poly.activate_interaction(self.current_mode == self.AnnotationMode.NONE)
                    self.annotations.append(poly); self.scene().addItem(poly)
            self.signal_proxy.annotation_updated.emit()
        except: pass
