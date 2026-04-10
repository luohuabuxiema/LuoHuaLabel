from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QToolBar, QDockWidget, QListWidget, QGraphicsView,
                               QLabel, QLineEdit, QPushButton, QStatusBar, QMenu)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QAction, QActionGroup, QPainter, QColor, QFont


class FormatSelectorWidget(QWidget):
    format_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 10)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignCenter)

        self.btn = QPushButton("JSON 格式 ▾")
        self.btn.setCursor(Qt.PointingHandCursor)

        self.btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                color: #333333;
                font-size: 14px;
                font-weight: bold;
                padding: 6px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
                color: #2196F3;
            }
            QPushButton::menu-indicator {
                image: none; /* 强制隐藏默认箭头 */
            }
        """)

        # 下拉菜单
        self.menu = QMenu(self)
        self.menu.setWindowFlag(Qt.FramelessWindowHint)
        self.menu.setAttribute(Qt.WA_TranslucentBackground)
        self.menu.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border: 1px solid #ebeef5;
                border-radius: 8px;
                padding: 6px 0px;
            }
            QMenu::item {
                padding: 8px 36px 8px 32px;
                margin: 2px 6px;
                border-radius: 4px;
                color: #606266;
                font-size: 13px;
                font-family: "Microsoft YaHei", sans-serif;
            }
            QMenu::item:selected {
                background-color: #ecf5ff;
                color: #409eff;
                font-weight: bold;
            }
        """)

        self.act_json = QAction("JSON 格式", self)
        self.act_yolo = QAction("YOLO 格式", self)
        self.act_xml = QAction("XML 格式", self)

        self.menu.addAction(self.act_json)
        self.menu.addAction(self.act_yolo)
        self.menu.addAction(self.act_xml)

        self.btn.setMenu(self.menu)

        self.label = QLabel("标注格式")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 11px; color: #888888; margin-top: 2px;")

        layout.addWidget(self.btn)
        layout.addWidget(self.label)

        self.act_json.triggered.connect(lambda: self._on_format_selected("json", "JSON 格式 ▾"))
        self.act_yolo.triggered.connect(lambda: self._on_format_selected("yolo", "YOLO 格式 ▾"))
        self.act_xml.triggered.connect(lambda: self._on_format_selected("xml", "XML 格式 ▾"))

    def _on_format_selected(self, fmt, text):
        self.btn.setText(text)
        self.format_changed.emit(fmt)

    def set_format(self, fmt):
        if fmt == "json":
            self.btn.setText("JSON 格式 ▾")
        elif fmt == "yolo":
            self.btn.setText("YOLO 格式 ▾")
        elif fmt == "xml":
            self.btn.setText("XML 格式 ▾")


class SwitchControl(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 26)
        self.setCursor(Qt.PointingHandCursor)
        self._checked = False

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked != checked:
            self._checked = checked
            self.toggled.emit(checked)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRect(0, 0, self.width(), self.height())

        if self._checked:
            p.setBrush(QColor("#2196F3"))
        else:
            p.setBrush(QColor("#BDBDBD"))

        p.setPen(Qt.NoPen)
        p.drawRoundedRect(rect, 13, 13)

        p.setBrush(QColor("#FFFFFF"))
        if self._checked:
            p.drawEllipse(self.width() - 24, 2, 22, 22)
        else:
            p.drawEllipse(2, 2, 22, 22)


class CanvasView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.NoDrag)

        self._is_panning = False
        self._pan_start_pos = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start_pos = event.position().toPoint()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("LuoHuaLabel - 基于SAM3的智能标注系统")
        MainWindow.resize(1280, 800)

        self.btnDatasetTool = QPushButton("数据集处理")
        self.btnDatasetTool.setStyleSheet("""
            QPushButton {
                background-color: #67c23a;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #85ce61; }
        """)

        self.centralWidget = QWidget(MainWindow)
        self.mainLayout = QHBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.view = CanvasView()
        self.mainLayout.addWidget(self.view)
        MainWindow.setCentralWidget(self.centralWidget)

        self.statusBar = QStatusBar()
        MainWindow.setStatusBar(self.statusBar)
        self.coordLabel = QLabel("坐标: X: 0, Y: 0")
        self.statusBar.addPermanentWidget(self.coordLabel)

        self.toolBar = QToolBar("工具栏")
        self.toolBar.setOrientation(Qt.Vertical)
        self.toolBar.setMovable(False)
        MainWindow.addToolBar(Qt.LeftToolBarArea, self.toolBar)

        self.dockRight = QDockWidget("标注管理", MainWindow)
        self.dockRight.setAllowedAreas(Qt.RightDockWidgetArea)
        self.dockRightWidget = QWidget()
        self.dockLayout = QVBoxLayout(self.dockRightWidget)

        self.labelClasses = QLabel("历史类别:")
        self.listClasses = QListWidget()

        self.labelFiles = QLabel("文件列表:")
        self.listFiles = QListWidget()
        # self.listFiles.setMinimumHeight(350)

        # self.dockLayout.addWidget(self.labelClasses)
        # self.dockLayout.addWidget(self.listClasses)
        # self.dockLayout.addWidget(self.labelFiles)
        # self.dockLayout.addWidget(self.listFiles)

        self.dockLayout.addWidget(self.labelClasses)
        self.dockLayout.addWidget(self.listClasses, 2)  # 历史类别分配x份自适应空间
        self.dockLayout.addWidget(self.labelFiles)
        self.dockLayout.addWidget(self.listFiles, 4)  # 文件列表分配x份自适应空间

        # self.dockLayout.addStretch()

        # ==== 右下角区域 ====
        self.samTextGroup = QWidget()
        textLayout = QVBoxLayout(self.samTextGroup)
        textLayout.setContentsMargins(0, 5, 0, 15)
        textLayout.setSpacing(8)

        helpLayout = QHBoxLayout()
        helpLayout.setContentsMargins(0, 0, 0, 8)
        helpLayout.addStretch()  # 把问号推到右边

        self.btnHelp = QPushButton("?")
        self.btnHelp.setToolTip("使用说明 (F1)")
        self.btnHelp.setFixedSize(22, 22)  # 尺寸
        self.btnHelp.setCursor(Qt.PointingHandCursor)

        font = QFont()
        font.setBold(True)
        font.setPointSize(10)  # 字体调小
        self.btnHelp.setFont(font)

        self.btnHelp.setStyleSheet("""
                    QPushButton {
                        background-color: transparent; /* 背景透明更柔和 */
                        border: 2px solid #dcdcdc;
                        border-radius: 11px; /* 完美圆形 */
                        color: #a0a0a0;
                        padding: 0px;
                        margin: 0px;
                    }
                    QPushButton:hover {
                        border: 2px solid #2196F3;
                        color: #2196F3;
                        background-color: #f0f8ff;
                    }
                    QPushButton:pressed {
                        background-color: #e3f2fd;
                    }
                """)
        helpLayout.addWidget(self.btnHelp)
        textLayout.addLayout(helpLayout)

        # 提示词输入与提取按钮
        self.samPromptInput = QLineEdit()
        self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog)")
        self.samPromptInput.setStyleSheet("""
                    QLineEdit {
                        border: 2px solid #e0e0e0;
                        border-radius: 14px;
                        padding: 6px 14px;
                        font-size: 13px;
                        background-color: #ffffff;
                        color: #333333;
                    }
                    QLineEdit:focus {
                        border: 2px solid #2196F3;
                    }
                """)

        self.samPromptBtn = QPushButton("✨ 提交")
        self.samPromptBtn.setCursor(Qt.PointingHandCursor)
        self.samPromptBtn.setStyleSheet("""
                    QPushButton {
                        background-color: #2196F3;
                        color: white;
                        border: none;
                        border-radius: 14px;
                        padding: 8px;
                        font-weight: bold;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: #1976D2;
                    }
                    QPushButton:pressed {
                        background-color: #0D47A1;
                    }
                """)

        textLayout.addWidget(self.samPromptInput)
        textLayout.addWidget(self.samPromptBtn)

        self.dockLayout.addWidget(self.samTextGroup)

        self.dockRight.setWidget(self.dockRightWidget)
        MainWindow.addDockWidget(Qt.RightDockWidgetArea, self.dockRight)

        self.actionOpen = QAction("打开目录", MainWindow)
        self.actionRect = QAction("矩形标注 (R)", MainWindow)
        self.actionPoly = QAction("多边形标注 (P)", MainWindow)
        self.actionPoint = QAction("点标注 (T)", MainWindow)
        self.actionRBox = QAction("旋转框标注 (O)", MainWindow)

        self.modeGroup = QActionGroup(MainWindow)
        for act in [self.actionRect, self.actionPoly, self.actionPoint, self.actionRBox]:
            act.setCheckable(True)
            self.modeGroup.addAction(act)

        self.actionRect.setChecked(True)

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addSeparator()

        self.formatWidget = FormatSelectorWidget()
        self.toolBar.addWidget(self.formatWidget)
        self.toolBar.addSeparator()

        self.toolBar.addAction(self.actionRect)
        self.toolBar.addAction(self.actionPoly)
        self.toolBar.addAction(self.actionPoint)
        self.toolBar.addAction(self.actionRBox)

        self.toolBar.addSeparator()
        self.samWidget = QWidget()
        samLayout = QVBoxLayout(self.samWidget)
        samLayout.setContentsMargins(5, 10, 5, 10)
        samLayout.setAlignment(Qt.AlignCenter)

        self.samSwitch = SwitchControl()
        self.samLabel = QLabel("SAM 智能辅助")
        self.samLabel.setAlignment(Qt.AlignCenter)
        self.samLabel.setStyleSheet("font-size: 11px; color: #555; margin-top: 5px;")

        samLayout.addWidget(self.samSwitch, alignment=Qt.AlignCenter)
        samLayout.addWidget(self.samLabel, alignment=Qt.AlignCenter)
        self.toolBar.addWidget(self.samWidget)

        # ==========================================
        # 数据集处理按钮
        # ==========================================
        self.toolBar.addSeparator()  # 添加一条水平分割线

        self.btnDatasetTool = QPushButton("数据集处理")
        self.btnDatasetTool.setCursor(Qt.PointingHandCursor)
        self.btnDatasetTool.setStyleSheet("""
                    QPushButton {
                        background-color: #f4f4f5;
                        color: #606266;
                        border: 1px solid #dcdfe6;
                        border-radius: 6px;
                        padding: 10px 5px;
                        margin: 10px 8px; /* 左右留边距，不贴死边框 */
                        font-weight: bold;
                        font-size: 13px;
                        font-family: "Microsoft YaHei";
                    }
                    QPushButton:hover {
                        background-color: #ecf5ff;
                        color: #409eff;
                        border-color: #c6e2ff;
                    }
                    QPushButton:pressed {
                        background-color: #3a8ee6;
                        color: white;
                    }
                """)
        self.toolBar.addWidget(self.btnDatasetTool)
