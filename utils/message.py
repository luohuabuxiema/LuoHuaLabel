# -*- coding: utf-8 -*-
"""
@Auth ： 落花不写码
@File ：message.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
@qq ：179958974
"""
import sys
import queue
# from PyQt5.QtWidgets import (
#     QApplication,
#     QWidget,
#     QPushButton, QHBoxLayout, QVBoxLayout,
# )
# from PyQt5.QtCore import (
#     QPoint,
#     Qt,
#     QTimer,
#     QPropertyAnimation,
#     QEasingCurve,
# )
# from PyQt5.QtGui import (
#     QPixmap,
#     QFont,
#     QPainter,
#     QPaintEvent,
#     QColor,
#     QFontMetrics, )

import queue
from PySide6.QtWidgets import QWidget, QGraphicsDropShadowEffect, QApplication, QPushButton, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import QPoint, Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPainter, QPaintEvent, QColor, QFontMetrics


class DialogOver(QWidget):
    _instanceWidget: queue.Queue = queue.Queue(7)
    _instanceDel: queue.Queue = queue.Queue(7)
    _instanceQueue: queue.Queue = queue.Queue(7)
    _count: list = [0, 0, 0, 0, 0, 0, 0]

    def __new__(cls, *args, **kwargs) -> None:
        try:
            _index = cls._count.index(0)
        except ValueError:
            return
        cls._count[_index] = 1
        cls._instanceWidget.put(_index)
        cls._instanceDel.put(_index)
        instance = super(DialogOver, cls).__new__(cls)
        cls._instanceQueue.put(instance)
        return instance

    def __del__(self) -> None:
        DialogOver._count[DialogOver._instanceDel.get()] = 0

    def __init__(self,
                 parent: QWidget,
                 text: str,
                 title: str = "",
                 flags: str = "info",
                 _showTime: int = 3000,
                 _dieTime: int = 500,
                 ):
        super().__init__()

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.title = title
        self.text = text
        self.flags = flags
        self.parent_widget = parent

        # 优化 1：配色体系（类似 Element UI）
        self.QBackgroundColor = QColor(240, 249, 235)
        self.QBorder = QColor(227, 249, 214)
        self.QTextColor = QColor(103, 194, 58)
        self.icon_text = "✅"

        self.w = 300
        self.h = 60
        self.resize(800, self.h + 20)  # 预留阴影空间
        self._dieTime = _dieTime

        # 优化 2：添加前端级别的高级阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(0, 4)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

        self.moveDialog()

        self.showTime = QTimer(self)
        self.showTime.setSingleShot(True)
        self.showTime.start(_showTime)
        self.showTime.timeout.connect(self.disDialog)

        self.dieTime = QTimer(self)
        self.dieTime.setSingleShot(True)
        self.dieTime.start(_showTime + _dieTime + 50)
        self.dieTime.timeout.connect(self.closeDialog)
        self.show()

    def paintStatus(self, flags):
        if flags == "success":
            self.QBackgroundColor = QColor(240, 249, 235)
            self.QBorder = QColor(227, 249, 214)
            self.QTextColor = QColor(103, 194, 58)
            self.icon_text = "✅"
        elif flags == "warning":
            self.QBackgroundColor = QColor(253, 246, 236)
            self.QBorder = QColor(250, 236, 216)
            self.QTextColor = QColor(230, 162, 60)
            self.icon_text = "⚠️"
        elif flags == "danger":
            self.QBackgroundColor = QColor(254, 240, 240)
            self.QBorder = QColor(253, 226, 226)
            self.QTextColor = QColor(245, 108, 108)
            self.icon_text = "❌"
        elif flags == "info":
            self.QBackgroundColor = QColor(237, 242, 252)
            self.QBorder = QColor(217, 236, 255)
            self.QTextColor = QColor(64, 158, 255)
            self.icon_text = "ℹ️"

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.paintStatus(self.flags)
        self.drawDialog(event, painter)

    def drawDialog(self, event: QPaintEvent, painter: QPainter) -> None:
        # 设置字体
        titleFont = QFont('Microsoft YaHei', 10, QFont.Bold)
        textFont = QFont('Microsoft YaHei', 9)
        iconFont = QFont('Segoe UI Emoji', 14)

        titleWidth = QFontMetrics(titleFont).horizontalAdvance(self.title)
        textWidth = QFontMetrics(textFont).horizontalAdvance(self.text)

        # 动态计算宽度
        self.w = max(250, 60 + max(titleWidth, textWidth) + 30)

        # 绘制背景矩形 (预留阴影边距)
        rect = event.rect()
        rect.setWidth(self.w)
        rect.setHeight(self.h)
        rect.translate(10, 10)  # 偏移以留出左侧和顶部的阴影

        painter.setPen(self.QBorder)
        painter.setBrush(self.QBackgroundColor)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # 画图标 (使用系统 Emoji)
        painter.setFont(iconFont)
        painter.setPen(self.QTextColor)
        painter.drawText(25, 10, 30, self.h, Qt.AlignVCenter | Qt.AlignLeft, self.icon_text)

        # 画标题
        painter.setFont(titleFont)
        painter.drawText(60, 18, titleWidth + 10, QFontMetrics(titleFont).height(), Qt.AlignLeft, self.title)

        # 画内容
        painter.setFont(textFont)
        painter.setPen(QColor(96, 98, 102))  # 文本颜色用深灰色
        painter.drawText(60, 38, textWidth + 10, QFontMetrics(textFont).height(), Qt.AlignLeft, self.text)

    def moveDialog(self) -> None:
        # 优化 3：定位在主窗口的右上方，并带有顺滑的从右侧滑入动画
        if self.parent_widget:
            # 获取主窗口在屏幕上的绝对坐标
            parent_pos = self.parent_widget.mapToGlobal(QPoint(0, 0))
            # X 坐标 = 主窗口右侧边缘 - 提示框宽度 - 边距
            x = parent_pos.x() + self.parent_widget.width() - self.w - 30
            # Y 坐标 = 主窗口顶部向下偏移 + 堆叠计算
            y = parent_pos.y() + 60 + (DialogOver._instanceWidget.get() * (self.h + 15))
        else:
            x, y = 100, 100

        animation = QPropertyAnimation(self, b"pos", self)
        # 从右侧 100 像素外滑入
        animation.setStartValue(QPoint(x + 100, y))
        animation.setEndValue(QPoint(x, y))
        animation.setDuration(600)
        animation.setEasingCurve(QEasingCurve.OutBack)
        animation.start()

    def disDialog(self) -> None:
        animation = QPropertyAnimation(self, b"windowOpacity", self)
        animation.setStartValue(1)
        animation.setEndValue(0)
        animation.setDuration(self._dieTime)
        animation.start()

    def closeDialog(self) -> None:
        DialogOver._instanceQueue.get()
        self.close()


class Window(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Window")
        self.resize(450, 650)

        layout = QVBoxLayout(self)
        buttonLayout = QHBoxLayout()

        button1 = QPushButton("success", self)
        button2 = QPushButton("warning", self)
        button3 = QPushButton("danger", self)
        button4 = QPushButton("info", self)

        buttonLayout.addWidget(button1)
        buttonLayout.addWidget(button2)
        buttonLayout.addWidget(button3)
        buttonLayout.addWidget(button4)

        layout.addStretch(1)
        layout.addLayout(buttonLayout)

        button1.clicked.connect(lambda x: self.dialog(title="success标题", text="success的内容", flags="success"))
        button2.clicked.connect(lambda x: self.dialog(title="warning标题", text="warning内容", flags="warning"))
        button3.clicked.connect(lambda x: self.dialog(title="danger标题", text="danger内容", flags="danger"))
        button4.clicked.connect(lambda x: self.dialog(title="info标题", text="info内容", flags="info"))

    def dialog(self, title, text, flags) -> None:
        DialogOver(parent=self, title=title, text=text, flags=flags)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Window()
    demo.show()
    sys.exit(app.exec())
