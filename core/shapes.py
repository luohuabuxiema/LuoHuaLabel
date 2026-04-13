from PySide6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem, \
    QGraphicsTextItem, QGraphicsLineItem, QGraphicsObject
from PySide6.QtGui import QPen, QBrush, QColor, QPolygonF, QFont, QTransform, QPainter
from PySide6.QtCore import Qt, QRectF, QPointF
import math


def point_to_segment_dist(p, a, b):
    px, py = p.x(), p.y()
    ax, ay = a.x(), a.y()
    bx, by = b.x(), b.y()

    ab_dist_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if ab_dist_sq == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5, a

    t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ab_dist_sq))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)

    dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
    return dist, QPointF(proj_x, proj_y)


def clamp_item_position(item, proposed_pos):
    scene = item.scene()
    if not scene: return proposed_pos
    rect = scene.sceneRect()
    shape_rect = item.boundingRect()

    min_x = rect.left() - shape_rect.left()
    max_x = rect.right() - shape_rect.right()
    min_y = rect.top() - shape_rect.top()
    max_y = rect.bottom() - shape_rect.bottom()

    new_x, new_y = proposed_pos.x(), proposed_pos.y()
    if min_x <= max_x:
        new_x = max(min_x, min(new_x, max_x))
    if min_y <= max_y:
        new_y = max(min_y, min(new_y, max_y))

    return QPointF(new_x, new_y)


class BaseShape:
    def setup_style(self, item):
        self.normal_pen = QPen(QColor(28, 126, 214), 2)
        self.normal_brush = QBrush(QColor(28, 126, 214, 50))
        self.hover_brush = QBrush(QColor(28, 126, 214, 120))

        item.setPen(self.normal_pen)
        item.setBrush(self.normal_brush)
        item.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        item.setAcceptHoverEvents(True)

    def apply_hover_enter(self, item):
        if not getattr(item, 'is_temp', False):
            item.setBrush(self.hover_brush)
            item.setCursor(Qt.PointingHandCursor)

    def apply_hover_leave(self, item):
        if not getattr(item, 'is_temp', False):
            item.setBrush(self.normal_brush)
            item.setCursor(Qt.ArrowCursor)

    def setup_label(self, item):
        self.label_text = QGraphicsTextItem(item)
        self.label_text.setDefaultTextColor(QColor(255, 255, 255))
        self.label_text.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_text.setZValue(1001)
        self.label_text.hide()

    def update_label_position(self, item):
        if not hasattr(self, 'label_text') or not self.label_text:
            return

        bound_rect = item.boundingRect()
        x = bound_rect.center().x()
        y = bound_rect.top() - 20
        self.label_text.setPos(x - self.label_text.boundingRect().width() / 2, y)

    def update_label_text(self, text):
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.setPlainText(text)
            self.label_text.show()

    def update_label_visibility(self, item, is_selected=False, is_hovered=False):
        if hasattr(self, 'label_text') and self.label_text:
            if is_selected or is_hovered:
                self.label_text.show()
            else:
                self.label_text.hide()


class HandleItem(QGraphicsEllipseItem):
    def __init__(self, parent, is_lt=False, is_rb=False):
        r = 3.5
        super().__init__(-r, -r, r * 2, r * 2, parent)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(QColor(28, 126, 214), 1.5))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(1000)
        self.hide()
        self._mouse_press_pos = None
        self._is_moved = False

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.SizeAllCursor)
        if hasattr(self.parentItem(), '_hovered'):
            self.parentItem()._hovered = True
            self.parentItem()._update_handle_visibility()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self._mouse_press_pos = event.pos()
        self._is_moved = False
        if self.parentItem():
            self.parentItem().setSelected(True)
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mouse_press_pos and (event.pos() - self._mouse_press_pos).manhattanLength() > 2:
            self._is_moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.parentItem():
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)

        if not self._is_moved and event.button() == Qt.LeftButton:
            parent = self.parentItem()
            if hasattr(parent, 'remove_handle'):
                parent.remove_handle(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() and self.parentItem():
            parent = self.parentItem()
            scene_pos = parent.mapToScene(value)
            rect = self.scene().sceneRect()
            clamped_x = max(rect.left(), min(scene_pos.x(), rect.right()))
            clamped_y = max(rect.top(), min(scene_pos.y(), rect.bottom()))
            return super().itemChange(change, parent.mapFromScene(QPointF(clamped_x, clamped_y)))

        if change == QGraphicsItem.ItemPositionHasChanged and self.scene():
            parent = self.parentItem()
            if hasattr(parent, 'update_from_handle') and not getattr(parent, '_updating_handles', False):
                parent.update_from_handle(self)
            elif hasattr(parent, 'update_from_handles') and not getattr(parent, '_updating_handles', False):
                parent.update_from_handles()
        return super().itemChange(change, value)


class RectShape(QGraphicsRectItem, BaseShape):
    def __init__(self, rect, label=""):
        super().__init__(rect)
        self.setup_style(self)
        self.label = label
        self._updating_handles = False
        self._hovered = False

        self.setup_label(self)
        if label:
            self.update_label_text(label)
            self.update_label_position(self)

        self.lt_handle = HandleItem(self)
        self.rt_handle = HandleItem(self)
        self.lb_handle = HandleItem(self)
        self.rb_handle = HandleItem(self)
        self.update_handles_pos()

    def hoverEnterEvent(self, event):
        self.apply_hover_enter(self)
        self._hovered = True
        self._update_handle_visibility()
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.apply_hover_leave(self)
        self._hovered = False
        self._update_handle_visibility()
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)

    def _update_handle_visibility(self):
        visible = self.isSelected() or self._hovered
        for h in [self.lt_handle, self.rt_handle, self.lb_handle, self.rb_handle]:
            h.setVisible(visible)

    def update_handles_pos(self):
        self._updating_handles = True
        r = self.rect()
        self.lt_handle.setPos(r.topLeft())
        self.rt_handle.setPos(r.topRight())
        self.lb_handle.setPos(r.bottomLeft())
        self.rb_handle.setPos(r.bottomRight())
        self._updating_handles = False

    def update_from_handle(self, dragged_handle):
        if self._updating_handles:
            return

        self._updating_handles = True
        hx, hy = dragged_handle.pos().x(), dragged_handle.pos().y()

        if dragged_handle == self.lt_handle:
            self.rt_handle.setPos(self.rt_handle.pos().x(), hy)
            self.lb_handle.setPos(hx, self.lb_handle.pos().y())
        elif dragged_handle == self.rt_handle:
            self.lt_handle.setPos(self.lt_handle.pos().x(), hy)
            self.rb_handle.setPos(hx, self.rb_handle.pos().y())
        elif dragged_handle == self.lb_handle:
            self.rb_handle.setPos(self.rb_handle.pos().x(), hy)
            self.lt_handle.setPos(hx, self.lt_handle.pos().y())
        elif dragged_handle == self.rb_handle:
            self.lb_handle.setPos(self.lb_handle.pos().x(), hy)
            self.rt_handle.setPos(hx, self.rt_handle.pos().y())

        min_x = min(self.lt_handle.pos().x(), self.rb_handle.pos().x())
        max_x = max(self.lt_handle.pos().x(), self.rb_handle.pos().x())
        min_y = min(self.lt_handle.pos().y(), self.rb_handle.pos().y())
        max_y = max(self.lt_handle.pos().y(), self.rb_handle.pos().y())

        self.setRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
        self.update_label_position(self)
        self._updating_handles = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False):
            return super().itemChange(change, clamp_item_position(self, value))

        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._update_handle_visibility()
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=self._hovered)
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)


class PolyShape(QGraphicsPolygonItem, BaseShape):
    def __init__(self, polygon, label="", is_temp=False):
        super().__init__(polygon)
        self.label = label
        self.is_temp = is_temp
        self.handles = []
        self._updating_handles = False
        self._hovered = False
        self._dragging_edge_idx = -1

        self.ghost_idx = -1
        self.ghost_pos = None
        self.ghost_handle = QGraphicsEllipseItem(-3.5, -3.5, 7, 7, self)
        self.ghost_handle.setBrush(QBrush(QColor(255, 255, 255, 180)))
        self.ghost_handle.setPen(QPen(QColor(28, 126, 214, 150), 1.5))
        self.ghost_handle.setZValue(999)
        self.ghost_handle.setAcceptedMouseButtons(Qt.NoButton)
        self.ghost_handle.hide()

        if is_temp:
            self.setPen(QPen(QColor(28, 126, 214), 2, Qt.DashLine))
            self.setBrush(QBrush(QColor(28, 126, 214, 50)))
        else:
            self.setup_style(self)
            self.setup_label(self)
            if label:
                self.update_label_text(label)
                self.update_label_position(self)

        self.update_handles()

    def hoverEnterEvent(self, event):
        self.apply_hover_enter(self)
        self._hovered = True
        self._update_handle_visibility()
        if not self.is_temp:
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        if self.is_temp or self._dragging_edge_idx != -1:
            super().hoverMoveEvent(event)
            return

        pt = event.pos()
        poly = self.polygon()
        min_dist = float('inf')
        insert_idx = -1
        closest_pt = None

        for i in range(poly.count()):
            p1 = poly[i]
            p2 = poly[(i + 1) % poly.count()]
            dist, proj = point_to_segment_dist(pt, p1, p2)
            if dist < min_dist:
                min_dist = dist
                insert_idx = i + 1
                closest_pt = proj

        if min_dist < 8:
            self.ghost_idx = insert_idx
            self.ghost_pos = closest_pt
            self.ghost_handle.setPos(closest_pt)
            self.ghost_handle.show()
            self.setCursor(Qt.CrossCursor)
        else:
            self.ghost_idx = -1
            self.ghost_handle.hide()
            self.setCursor(Qt.PointingHandCursor)

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.apply_hover_leave(self)
        self._hovered = False
        self.ghost_idx = -1
        self.ghost_handle.hide()

        if not any(h.isUnderMouse() for h in self.handles):
            self._update_handle_visibility()

        if not self.is_temp:
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.ghost_idx != -1:
            self._dragging_edge_idx = self.ghost_idx
            poly = self.polygon()
            poly.insert(self.ghost_idx, self.ghost_pos)
            self.setPolygon(poly)
            self.ghost_handle.hide()
            self.ghost_idx = -1

            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_edge_idx != -1:
            poly = self.polygon()

            scene_pos = self.mapToScene(event.pos())
            scene = self.scene()
            if scene:
                rect = scene.sceneRect()
                clamped_x = max(rect.left(), min(scene_pos.x(), rect.right()))
                clamped_y = max(rect.top(), min(scene_pos.y(), rect.bottom()))
                clamped_local = self.mapFromScene(QPointF(clamped_x, clamped_y))
                poly[self._dragging_edge_idx] = clamped_local
            else:
                poly[self._dragging_edge_idx] = event.pos()

            self.setPolygon(poly)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_edge_idx != -1:
            self._dragging_edge_idx = -1
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def remove_handle(self, handle):
        if len(self.handles) <= 3:
            return
        idx = self.handles.index(handle)
        poly = self.polygon()
        poly.remove(idx)
        self.setPolygon(poly)

    def _update_handle_visibility(self):
        if self.is_temp: return
        visible = self.isSelected() or self._hovered
        for h in self.handles:
            h.setVisible(visible)

    def update_handles(self):
        polygon = self.polygon()
        while len(self.handles) < polygon.count():
            handle = HandleItem(self)
            if self.is_temp:
                handle.setFlag(QGraphicsItem.ItemIsMovable, False)
                handle.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
                handle.hide()
            self.handles.append(handle)

        while len(self.handles) > polygon.count():
            h = self.handles.pop()
            h.setParentItem(None)
            if self.scene(): self.scene().removeItem(h)

        self._updating_handles = True
        for i, handle in enumerate(self.handles):
            handle.setPos(polygon[i])
        self._update_handle_visibility()
        self._updating_handles = False

    def setPolygon(self, polygon):
        super().setPolygon(polygon)
        self.update_handles()
        if not self.is_temp:
            self.update_label_position(self)

    def update_from_handles(self):
        if self.is_temp or self._updating_handles: return
        polygon = QPolygonF()
        for handle in self.handles:
            polygon.append(handle.pos())
        super().setPolygon(polygon)
        self.update_label_position(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False):
            return super().itemChange(change, clamp_item_position(self, value))

        if not self.is_temp:
            if change == QGraphicsItem.ItemSelectedHasChanged:
                self._update_handle_visibility()
                self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=self._hovered)
            elif change == QGraphicsItem.ItemPositionHasChanged:
                self.update_label_position(self)
        return super().itemChange(change, value)


class PointShape(QGraphicsEllipseItem, BaseShape):
    def __init__(self, point, label=""):
        r = 4
        super().__init__(point.x() - r, point.y() - r, r * 2, r * 2)
        self.normal_pen = QPen(QColor(250, 82, 82), 2)
        self.normal_brush = QBrush(QColor(250, 82, 82, 150))
        self.hover_brush = QBrush(QColor(250, 82, 82, 220))
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.label = label

        self.setup_label(self)
        if label:
            self.update_label_text(label)
            self.update_label_position(self)

    def hoverEnterEvent(self, event):
        self.setBrush(self.hover_brush)
        self.setCursor(Qt.PointingHandCursor)
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(self.normal_brush)
        self.setCursor(Qt.ArrowCursor)
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return super().itemChange(change, clamp_item_position(self, value))

        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=self.isUnderMouse())
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)


class OBBHandle(QGraphicsItem):
    """自定义胶囊形/圆形的 OBB 操作手柄 (接管底层鼠标事件)"""

    def __init__(self, handle_type, parent):
        super().__init__(parent)
        self.handle_type = handle_type  # 'top', 'bottom', 'left', 'right', 'rotate'
        self.setAcceptHoverEvents(True)
        self.setZValue(100)

        self.w, self.h = 0, 0
        if self.handle_type in ['top', 'bottom']:
            self.w, self.h = 16, 6  # 横向胶囊
        elif self.handle_type in ['left', 'right']:
            self.w, self.h = 6, 16  # 纵向胶囊
        elif self.handle_type == 'rotate':
            self.w, self.h = 10, 10  # 旋转圆球

    def boundingRect(self):
        return QRectF(-self.w / 2, -self.h / 2, self.w, self.h)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(28, 126, 214), 2))
        if self.handle_type == 'rotate':
            painter.drawEllipse(self.boundingRect())
        else:
            painter.drawRoundedRect(self.boundingRect(), min(self.w, self.h) / 2, min(self.w, self.h) / 2)

    def hoverEnterEvent(self, event):
        if self.handle_type == 'rotate':
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    # ================= 鼠标事件接管 =================
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.handle_type == 'rotate':
                self.setCursor(Qt.ClosedHandCursor)

            # 记录当前正在拖拽的手柄，并锁定父容器不被整体拖走
            self.parentItem()._dragging_handle = self.handle_type
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.parentItem()._dragging_handle == self.handle_type:
            # 将拖拽产生的全局坐标，实时发送给父容器进行解算
            self.parentItem().handle_dragged(self.handle_type, event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parentItem()._dragging_handle = None
            # 解除父容器锁定
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)
            if self.handle_type == 'rotate':
                self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class RotatedRectShape(QGraphicsObject, BaseShape):
    def __init__(self, cx, cy, w, h, angle, label="", is_temp=False):
        super().__init__()
        self.is_temp = is_temp
        self.label = label

        # 用于区分“整体移动”和“手柄拉伸”
        self._is_resizing = False

        # 依赖宽高
        self.box_w = w
        self.box_h = h
        self.setPos(cx, cy)
        self.setRotation(angle)

        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        self._dragging_handle = None
        self._hovered = False

        self.rect_item = QGraphicsRectItem(self)
        if is_temp:
            self.rect_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            self.rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
        else:
            self.rect_item.setPen(QPen(QColor(28, 126, 214), 2))
            self.rect_item.setBrush(QBrush(QColor(28, 126, 214, 50)))

        self.rotate_line = QGraphicsLineItem(self)
        self.rotate_line.setPen(QPen(QColor(28, 126, 214), 1.5, Qt.DashLine))

        self.h_top = OBBHandle('top', self)
        self.h_bottom = OBBHandle('bottom', self)
        self.h_left = OBBHandle('left', self)
        self.h_right = OBBHandle('right', self)
        self.h_rotate = OBBHandle('rotate', self)

        self.handles = [self.h_top, self.h_bottom, self.h_left, self.h_right, self.h_rotate, self.rotate_line]

        self.update_geometry()
        self._update_handle_visibility()

        if not is_temp:
            self.setup_label(self)
            if label: self.update_label_text(label)

        if is_temp:
            for h in self.handles:
                h.hide()

    def boundingRect(self):
        r = self.box_h / 2 + 35
        return QRectF(-self.box_w / 2 - 10, -r, self.box_w + 20, r + self.box_h / 2 + 10)

    def paint(self, painter, option, widget=None):
        pass

    def update_geometry(self):
        w, h = self.box_w, self.box_h
        self.rect_item.setRect(-w / 2, -h / 2, w, h)

        self.h_top.setPos(0, -h / 2)
        self.h_bottom.setPos(0, h / 2)
        self.h_left.setPos(-w / 2, 0)
        self.h_right.setPos(w / 2, 0)

        self.h_rotate.setPos(0, -h / 2 - 30)
        self.rotate_line.setLine(0, -h / 2, 0, -h / 2 - 30)

        self.prepareGeometryChange()
        self.update_label_position(self)

    def handle_dragged(self, handle_type, scene_pos):
        self._is_resizing = True

        local_pos = self.mapFromScene(scene_pos)
        if handle_type == 'top':
            dy = local_pos.y() - (-self.box_h / 2)
            if self.box_h - dy < 5: dy = self.box_h - 5
            self.box_h -= dy
            scene_offset = self.mapToScene(QPointF(0, dy / 2)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'bottom':
            dy = local_pos.y() - (self.box_h / 2)
            if self.box_h + dy < 5: dy = -(self.box_h - 5)
            self.box_h += dy
            scene_offset = self.mapToScene(QPointF(0, dy / 2)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'left':
            dx = local_pos.x() - (-self.box_w / 2)
            if self.box_w - dx < 5: dx = self.box_w - 5
            self.box_w -= dx
            scene_offset = self.mapToScene(QPointF(dx / 2, 0)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'right':
            dx = local_pos.x() - (self.box_w / 2)
            if self.box_w + dx < 5: dx = -(self.box_w - 5)
            self.box_w += dx
            scene_offset = self.mapToScene(QPointF(dx / 2, 0)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'rotate':
            center_scene = self.mapToScene(QPointF(0, 0))
            dx = scene_pos.x() - center_scene.x()
            dy = scene_pos.y() - center_scene.y()
            angle_deg = math.degrees(math.atan2(dy, dx))
            self.setRotation(angle_deg + 90)

        self.update_geometry()
        self._is_resizing = False

    def polygon(self):
        w, h = self.box_w, self.box_h
        pts = [
            QPointF(-w / 2, -h / 2),
            QPointF(w / 2, -h / 2),
            QPointF(w / 2, h / 2),
            QPointF(-w / 2, h / 2)
        ]
        return QPolygonF([self.mapToScene(p) for p in pts])

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.rect_item.setBrush(QBrush(QColor(28, 126, 214, 120)))
        self._update_handle_visibility()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.rect_item.setBrush(QBrush(QColor(28, 126, 214, 50)))
        self._update_handle_visibility()
        super().hoverLeaveEvent(event)

    def _update_handle_visibility(self):
        if self.is_temp: return
        visible = self.isSelected() or self._hovered
        for h in self.handles:
            h.setVisible(visible)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False) and not getattr(self, '_is_resizing', False):
            scene = self.scene()
            if scene:
                rect = scene.sceneRect()
                new_pos = value
                valid_x = max(rect.left(), min(new_pos.x(), rect.right()))
                valid_y = max(rect.top(), min(new_pos.y(), rect.bottom()))
                return QPointF(valid_x, valid_y)

        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._update_handle_visibility()
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)