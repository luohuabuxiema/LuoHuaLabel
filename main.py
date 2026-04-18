import sys
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox, QLabel, QListWidgetItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPolygonF

from main_dataset_tool import DatasetToolWindow
from ui.main_window import Ui_MainWindow
from core.canvas import Canvas, CanvasMode
from core.sam_client import SAMClient
from core.exporter import Exporter
from core.shapes import RectShape, PolyShape, PointShape, RotatedRectShape
from utils.message import DialogOver


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        with open(os.path.join(os.path.dirname(__file__), "ui/style.qss"), "r", encoding="utf-8") as f:
            self.setStyleSheet(f.read())

        self.scene = Canvas(self)
        self.view.setScene(self.scene)

        self.current_image_path = None
        self.current_dir = None
        self.class_list = []
        self.current_format = "json"

        self.modeLabel = QLabel("模式: 矩形标注")
        self.statusBar.addWidget(self.modeLabel)

        self.helpLabel = QLabel("状态: 正在初始化")
        self.statusBar.addWidget(self.helpLabel)

        self.sam_client = SAMClient(self)
        self.sam_client.inference_result.connect(self.scene.handle_sam_result)
        self.sam_client.text_result_ready.connect(self.handle_text_results)
        self.sam_client.model_status_changed.connect(self.update_model_status)
        self.scene.sam_client = self.sam_client

        # 撤销/重做时数据栈
        self.undo_stack = []
        self.redo_stack = []
        self.max_history_steps = 20  # 保留20步历史记录
        self.scene.state_changed.connect(self.push_state)  # 绑定画板信号

        self._connect_signals()
        self._set_mode(CanvasMode.RECT)
        self.sam_client.load_model_async(r"E:\2-浏览器下载的文件\sam3.pt")  # 模型路径

    def _connect_signals(self):
        self.actionOpen.triggered.connect(self.open_dir)

        # 下拉菜单组件信号
        self.formatWidget.format_changed.connect(self.set_current_format)

        self.btnDatasetTool.clicked.connect(self.open_dataset_tool)

        # self.actionFormatJSON.triggered.connect(lambda: self.set_current_format("json"))
        # self.actionFormatYOLO.triggered.connect(lambda: self.set_current_format("yolo"))
        # self.actionFormatXML.triggered.connect(lambda: self.set_current_format("xml"))

        self.actionRect.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.RECT))
        self.actionPoly.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POLY))
        self.actionPoint.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POINT))
        self.actionRBox.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.RBOX))

        self.samSwitch.toggled.connect(self.on_sam_toggled)

        self.samPromptBtn.clicked.connect(self.trigger_sam_prompt)
        self.samPromptInput.returnPressed.connect(self.trigger_sam_prompt)

        self.listFiles.currentItemChanged.connect(self.on_file_selected)
        self.scene.mouse_moved.connect(self.update_coordinate_label)
        self.scene.shape_drawn.connect(self.handle_new_shape)

        self.scene.shape_double_clicked.connect(self.edit_shape_label)  # 双击修改

        self.listClasses.itemChanged.connect(self.on_list_item_changed)

        self.btnHelp.clicked.connect(self.show_help_dialog)

    def add_class_to_list(self, cls_name):
        """列表项支持双击编辑"""
        if cls_name not in self.class_list:
            self.class_list.append(cls_name)
            item = QListWidgetItem(cls_name)
            # 开启双击编辑权限
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, cls_name)
            self.listClasses.addItem(item)

    def push_state(self):
        """把当前画布状态拍个快照，存进撤销堆栈"""
        if not self.current_image_path: return
        current_state = Exporter.extract_shapes(self.scene)

        # 如果拖了一下鼠标但什么都没变，就不存，节约内存
        if self.undo_stack:
            last_state = self.undo_stack[-1]
            if json.dumps(last_state, sort_keys=True) == json.dumps(current_state, sort_keys=True):
                return

        self.undo_stack.append(current_state)
        # 限制最大步数
        if len(self.undo_stack) > self.max_history_steps:
            self.undo_stack.pop(0)

        # 一旦有新操作，重做（前进）堆栈必须清空
        self.redo_stack.clear()

    def undo(self):
        """撤销 (Ctrl+Z)"""
        if len(self.undo_stack) > 1:
            # 把现在的状态拿出来，放到重做栈里去
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)
            # 获取上一步的状态并还原
            previous_state = self.undo_stack[-1]
            self.restore_state(previous_state)

    def redo(self):
        """重做/前进 (Ctrl+Y 或 Ctrl+Shift+Z)"""
        if self.redo_stack:
            # 从重做栈里拿出来，塞回撤销栈
            next_state = self.redo_stack.pop()
            self.undo_stack.append(next_state)
            # 还原该状态
            self.restore_state(next_state)

    def restore_state(self, state):
        """根据快照数据，完全重建画板元素"""
        self.scene.clear_shapes()
        for shape_data in state:
            label = shape_data.get("label", "")
            shape_type = shape_data.get("type", "")
            points = shape_data.get("points", [])

            # 同步类别列表
            if label and label not in self.class_list:
                # self.class_list.append(label)
                # self.listClasses.addItem(label)
                self.add_class_to_list(label)
                self.save_classes()

            shape = None
            if shape_type == "rectangle" and len(points) == 2:
                rect = QRectF(points[0][0], points[0][1], points[1][0] - points[0][0], points[1][1] - points[0][1])
                shape = RectShape(rect, label)
            elif shape_type == "polygon" and len(points) >= 3:
                qpoints = [QPointF(p[0], p[1]) for p in points]
                shape = PolyShape(QPolygonF(qpoints), label)
            elif shape_type == "point" and len(points) == 1:
                shape = PointShape(QPointF(points[0][0], points[0][1]), label)
            elif shape_type == "obb":
                rect_data = shape_data.get("rect")
                angle = shape_data.get("angle", 0)
                if rect_data and len(rect_data) == 4:
                    cx, cy, w, h = rect_data[0], rect_data[1], rect_data[2], rect_data[3]
                    shape = RotatedRectShape(cx, cy, w, h, angle, label)

            if shape:
                self.scene.addItem(shape)
                if hasattr(shape, 'update_label_text'):
                    shape.update_label_text(label)
                if hasattr(shape, 'update_label_position'):
                    shape.update_label_position(shape)
                if hasattr(shape, 'update_label_visibility'):
                    shape.update_label_visibility(shape, is_selected=False, is_hovered=False)

        self.auto_save_annotation()

    def open_dataset_tool(self):
        try:
            if not hasattr(self, 'dataset_window') or self.dataset_window is None:
                self.dataset_window = DatasetToolWindow()
            # 显示窗口
            self.dataset_window.show()
            # 把窗口强制拉到最前面
            self.dataset_window.raise_()
            self.dataset_window.activateWindow()

        except Exception as e:
            DialogOver(self, f"启动失败: {e}", "系统错误", "danger")

    def trigger_sam_prompt(self):
        if self.scene.mode == CanvasMode.POINT:
            DialogOver(self, "点标注模式下无法使用 SAM 智能提取", "提示", "warning")
            return

        prompt = self.samPromptInput.text().strip()
        if prompt:
            self.samSwitch.setChecked(True)
            self.helpLabel.setText(f"正在提取提示词: {prompt}...")
            self.helpLabel.setStyleSheet("color: orange;")
            self.sam_client.request_text_inference(prompt)

    def handle_text_results(self, results, prompt_text):
        if not results:
            self.helpLabel.setText(f"提取完成: 未发现关于 '{prompt_text}' 的目标")
            self.helpLabel.setStyleSheet("color: red;")
            return

        self.helpLabel.setText(f"提取完成: 成功抓取 {len(results)} 个 '{prompt_text}' 目标")
        self.helpLabel.setStyleSheet("color: green;")

        if prompt_text not in self.class_list:
            # self.class_list.append(prompt_text)
            # self.listClasses.addItem(prompt_text)
            self.add_class_to_list(prompt_text)
            self.save_classes()

        for res in results:
            if self.scene.mode == CanvasMode.RECT:
                x, y, w, h = res["rect"]
                shape = RectShape(QRectF(x, y, w, h), prompt_text)
            else:
                qpts = [QPointF(p[0], p[1]) for p in res["poly_pts"]]
                shape = PolyShape(QPolygonF(qpts), prompt_text)

            self.scene.addItem(shape)
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(prompt_text)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            if hasattr(shape, 'update_label_visibility'):
                shape.update_label_visibility(shape, is_selected=False, is_hovered=False)

        self.auto_save_annotation()

    def show_help_dialog(self):
        help_text = """
        <h3>【快捷键大全】</h3>
        <ul>
            <li><b>A / 左方向键</b>：上一张图片</li>
            <li><b>D / 右方向键</b>：下一张图片</li>
            <li><b style="color:red;">Ctrl + S</b>：保存当前标注</li>
            <li><b style="color:blue;">Q</b>：开启/关闭 SAM 智能辅助</li>
            <li><b>R</b>：切换至 矩形标注</li>
            <li><b>P</b>：切换至 多边形标注</li>
            <li><b>T</b>：切换至 点标注</li>
            <li><b>O</b>：切换至 旋转框标注</li>
            <li><b>Del / Backspace</b>：删除当前选中的标注框</li>
            <li><b>F1</b>：打开此帮助文档</li>
        </ul>
        <hr>
        <h3>【多边形绘制技巧】</h3>
        <ul>
            <li><b>左键点击</b>：添加顶点</li>
            <li><b>Ctrl + Z</b>：撤销上一个顶点</li>
            <li><b>双击 / Enter</b>：闭合多边形</li>

        </ul>
        <hr>
        <h3>【旋转框绘制快捷键】</h3>
        <ul>
            <li><b> Z / V</b>：每次向左/向右旋转 5°</li>
            <li><b>X / C</b>：每次向左/向右旋转 1°</li>
        </ul>
        <hr>
        <h3>【SAM 智能辅助】</h3>
        <ul>
            <li><b>鼠标点选</b>：开启开关后，鼠标悬停预览，点击直接确认生成高精度轮廓。</li>
            <li><b>提示词提取</b>：在右下角输入框输入目标名称（如: dog），按回车即可一键全图抓取并打好框！左侧选中的是“矩形”还是“多边形”格式。</li>
        </ul>
        """
        QMessageBox.about(self, "LuoHuaLabel 使用说明", help_text)

    def update_coordinate_label(self, x, y):
        self.coordLabel.setText(f"坐标: X: {x}, Y: {y}")

    def on_sam_toggled(self, checked):
        self.scene.set_sam_enabled(checked)
        self._update_help_text(self.scene.mode)

    def _set_mode(self, mode):
        self.scene.set_mode(mode)
        mode_name = CanvasMode.get_mode_name(mode)
        self.modeLabel.setText(f"模式: {mode_name}标注")
        self._update_help_text(mode)

        if mode == CanvasMode.RECT:
            self.actionRect.setChecked(True)
        elif mode == CanvasMode.POLY:
            self.actionPoly.setChecked(True)
        elif mode == CanvasMode.POINT:
            self.actionPoint.setChecked(True)
        elif mode == CanvasMode.RBOX:
            self.actionRBox.setChecked(True)

        if mode == CanvasMode.POINT:
            if self.samSwitch.isChecked():
                self.samSwitch.setChecked(False)

            self.samSwitch.setEnabled(False)
            self.samPromptInput.setEnabled(False)
            self.samPromptBtn.setEnabled(False)
            self.samPromptInput.setPlaceholderText("点标注模式下 SAM 不可用")
        else:
            self.samSwitch.setEnabled(True)
            self.samPromptInput.setEnabled(True)
            self.samPromptBtn.setEnabled(True)
            self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog)")

    def _update_help_text(self, mode):
        is_sam = self.samSwitch.isChecked()
        if mode == CanvasMode.RECT:
            if is_sam:
                self.helpLabel.setText("操作: 鼠标悬停实时预览外接矩形，左键点击直接确认生成矩形框")
            else:
                self.helpLabel.setText("操作: 拖动鼠标绘制常规矩形")
        elif mode == CanvasMode.POLY:
            if is_sam:
                self.helpLabel.setText("操作: 鼠标悬停实时预览轮廓节点，左键点击直接确认生成多边形")
            else:
                self.helpLabel.setText("操作: 点击添加顶点，双击闭合多边形")
        elif mode == CanvasMode.POINT:
            self.helpLabel.setText("操作: 点击添加点标注")
        elif mode == CanvasMode.RBOX:
            self.helpLabel.setText("操作: 拖动绘制旋转框，Z/X/C/V调整角度")

    def load_classes(self, dir_path):
        self.class_list.clear()
        self.listClasses.clear()
        class_file = os.path.join(dir_path, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                for line in f:
                    cls_name = line.strip()
                    if cls_name:
                        self.add_class_to_list(cls_name)
                        # self.class_list.append(cls_name)
                        # self.listClasses.addItem(cls_name)

    def save_classes(self):
        if self.current_dir:
            class_file = os.path.join(self.current_dir, "classes.txt")
            with open(class_file, "w", encoding="utf-8") as f:
                for cls_name in self.class_list:
                    f.write(cls_name + "\n")

    def handle_new_shape(self, shape):
        self.scene.addItem(shape)
        QApplication.processEvents()

        last_class = self.class_list[-1] if self.class_list else ""
        default_idx = self.class_list.index(last_class) if last_class in self.class_list else 0

        cls_name, ok = QInputDialog.getItem(self, "输入类别", "请选择或输入类别名称:", self.class_list, default_idx,
                                            True)

        if ok and cls_name:
            cls_name = cls_name.strip()
            if cls_name not in self.class_list:
                # self.class_list.append(cls_name)
                # self.listClasses.addItem(cls_name)
                self.add_class_to_list(cls_name)
                self.save_classes()

            shape.label = cls_name
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(cls_name)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            if hasattr(shape, 'update_label_visibility'):
                shape.update_label_visibility(shape, is_selected=True, is_hovered=False)
            for item in self.scene.selectedItems():
                item.setSelected(False)
            shape.setSelected(True)
            self.push_state()
        else:
            self.scene.removeItem(shape)

    def edit_shape_label(self, shape):
        """二次修改已有标注框的类别"""
        current_label = shape.label
        default_idx = self.class_list.index(current_label) if current_label in self.class_list else 0

        cls_name, ok = QInputDialog.getItem(self, "修改类别", "请重新选择或输入类别名称:", self.class_list, default_idx,
                                            True)

        if ok and cls_name:
            cls_name = cls_name.strip()
            if cls_name not in self.class_list:
                # self.class_list.append(cls_name)
                # self.listClasses.addItem(cls_name)
                self.add_class_to_list(cls_name)
                self.save_classes()

            # 更新形状的数据和标签显示
            shape.label = cls_name
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(cls_name)

            # 修改完自动保存一次
            self.auto_save_annotation()
            self.push_state()

    def on_list_item_changed(self, item):
        """处理右侧列表双击修改类别名的全局涟漪效应"""
        new_name = item.text().strip()
        old_name = item.data(Qt.UserRole)

        # 如果没有真正改动，直接跳过
        if not old_name or new_name == old_name:
            return

        self.listClasses.blockSignals(True)
        try:
            if not new_name:
                DialogOver(self, "类别名不能为空！", "名称错误", "warning")
                item.setText(old_name)
                return

            if new_name in self.class_list:
                DialogOver(self, f"类别名 '{new_name}' 已存在！", "名称冲突", "warning")
                item.setText(old_name)
                return

            # 替换内部字典
            idx = self.class_list.index(old_name)
            self.class_list[idx] = new_name
            item.setData(Qt.UserRole, new_name)  # 把新名字设为基准

            # 遍历画板，把所有旧名字的框换成新名字
            changed = False
            for shape in self.scene.items():
                if isinstance(shape, (RectShape, PolyShape, PointShape, RotatedRectShape)):
                    if getattr(shape, 'label', '') == old_name:
                        shape.label = new_name
                        if hasattr(shape, 'update_label_text'):
                            shape.update_label_text(new_name)
                        changed = True

            # 保存并推入时光机
            self.save_classes()
            if changed:
                self.auto_save_annotation()
                self.push_state()

            DialogOver(self, f"已将所有的 '{old_name}' 批量变更为 '{new_name}'", "修改成功", "success")

        finally:
            self.listClasses.blockSignals(False)

    def open_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.current_dir = dir_path
            self.listFiles.clear()
            self.load_classes(dir_path)
            for f in os.listdir(dir_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.listFiles.addItem(os.path.join(dir_path, f))

            if self.listFiles.count() > 0:
                self.listFiles.setCurrentRow(0)

    # def update_model_status(self, success, msg):
    #     self.helpLabel.setText(msg)
    #     if success:
    #         self.helpLabel.setStyleSheet("color: green;")
    #     else:
    #         self.helpLabel.setStyleSheet("color: red;")

    def update_model_status(self, success, msg):
        self.helpLabel.setText(msg)
        if success:
            self.helpLabel.setStyleSheet("color: green;")
            # 模型加载成功后，检查用户是不是已经提前打开图片了
            if self.current_image_path:
                self.helpLabel.setText("模型已就绪，正在自动分析当前图片特征...")
                self.helpLabel.setStyleSheet("color: orange;")
                QApplication.processEvents()
                self.sam_client.set_image(self.current_image_path)
                self.helpLabel.setText("分析完成，可以开始智能标注")
                self.helpLabel.setStyleSheet("color: green;")
        else:
            self.helpLabel.setStyleSheet("color: red;")

    def on_file_selected(self, current, previous):
        if previous:
            self.auto_save_annotation()

        if current:
            path = current.text()
            self.current_image_path = path
            self.scene.load_image(path)
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.load_annotations(path)

            self.undo_stack.clear()
            self.redo_stack.clear()
            self.push_state()

            if self.sam_client.model:
                self.helpLabel.setText("正在分析图片智能特征...")
                self.helpLabel.setStyleSheet("color: orange;")
                QApplication.processEvents()
                self.sam_client.set_image(path)
                self.helpLabel.setText("分析完成，可以开始智能标注")
                self.helpLabel.setStyleSheet("color: green;")
            else:
                self.helpLabel.setText("等待后台加载模型，稍后将自动分析图片...")
                self.helpLabel.setStyleSheet("color: orange;")

    def auto_save_annotation(self):
        if not self.current_image_path or not self.scene.img_item: return
        shapes_data = Exporter.extract_shapes(self.scene)
        # if not shapes_data: return

        img_rect = self.scene.img_item.pixmap().rect()
        base_name = os.path.splitext(self.current_image_path)[0]

        try:
            if self.current_format == "json":
                out_path = base_name + ".json"
                Exporter.save_json(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)
            elif self.current_format == "yolo":
                out_path = base_name + ".txt"
                Exporter.save_yolo(out_path, img_rect.width(), img_rect.height(), shapes_data, self.class_list)
            elif self.current_format == "xml":
                out_path = base_name + ".xml"
                Exporter.save_xml(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)
        except Exception as e:
            print(f"自动保存失败: {str(e)}")

    def set_current_format(self, format_type):
        self.current_format = format_type
        self.formatWidget.set_format(format_type)

        # if format_type == "json":
        #     self.actionFormatJSON.setChecked(True)
        # elif format_type == "yolo":
        #     self.actionFormatYOLO.setChecked(True)
        # elif format_type == "xml":
        #     self.actionFormatXML.setChecked(True)

        if self.current_image_path:
            self.scene.clear_shapes()
            self.load_annotations(self.current_image_path)
        DialogOver(self, f"当前保存及读取格式变为 {format_type.upper()}", "格式切换", "info")

    def load_annotations(self, image_path):
        if not self.scene.img_item: return

        img_w = self.scene.img_item.pixmap().width()
        img_h = self.scene.img_item.pixmap().height()
        base_path = os.path.splitext(image_path)[0]

        if self.current_format == "json":
            self._load_json(base_path + ".json")
        elif self.current_format == "yolo":
            self._load_yolo(base_path + ".txt", img_w, img_h)
        elif self.current_format == "xml":
            self._load_xml(base_path + ".xml")

    def _add_shape_to_scene(self, shape, label):
        """往画板内添加加载出来的轮廓并同步历史类别"""
        if label not in self.class_list:
            # self.class_list.append(label)
            # self.listClasses.addItem(label)
            self.add_class_to_list(label)
            self.save_classes()
        self.scene.addItem(shape)

    def _load_json(self, json_path):
        if not os.path.exists(json_path): return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for shape_data in data.get("shapes", []):
                label = shape_data.get("label", "")
                points = shape_data.get("points", [])
                shape_type = shape_data.get("shape_type", "rectangle")

                if shape_type == "rectangle" and len(points) == 2:
                    rect = QRectF(points[0][0], points[0][1], points[1][0] - points[0][0], points[1][1] - points[0][1])
                    shape = RectShape(rect, label)
                elif shape_type == "polygon" and len(points) >= 3:
                    qpoints = [QPointF(p[0], p[1]) for p in points]
                    shape = PolyShape(QPolygonF(qpoints), label)
                elif shape_type == "point" and len(points) == 1:
                    shape = PointShape(QPointF(points[0][0], points[0][1]), label)
                # 旋转框 (OBB) 的解析分支
                elif shape_type == "obb":
                    rect_data = shape_data.get("rect")
                    angle = shape_data.get("angle", 0)
                    if rect_data and len(rect_data) == 4:
                        cx, cy, w, h = rect_data[0], rect_data[1], rect_data[2], rect_data[3]
                        shape = RotatedRectShape(cx, cy, w, h, angle, label)
                    else:
                        continue
                else:
                    continue
                self._add_shape_to_scene(shape, label)
        except Exception as e:
            print(f"加载 JSON 标注失败: {e}")

    def _load_yolo(self, txt_path, img_w, img_h):
        if not os.path.exists(txt_path): return
        import math  # 局部导入数学库，用于逆向推导
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                class_id = int(parts[0])
                label = self.class_list[class_id] if class_id < len(self.class_list) else str(class_id)

                # 1. YOLO BBox 格式 (常规矩形：5个参数)
                if len(parts) == 5:
                    cx, cy = float(parts[1]) * img_w, float(parts[2]) * img_h
                    w, h = float(parts[3]) * img_w, float(parts[4]) * img_h
                    shape = RectShape(QRectF(cx - w / 2, cy - h / 2, w, h), label)

                # YOLO OBB 旋转框格式 (9个参数：1 个类别 + 8 个坐标)
                elif len(parts) == 9:
                    x1, y1 = float(parts[1]) * img_w, float(parts[2]) * img_h
                    x2, y2 = float(parts[3]) * img_w, float(parts[4]) * img_h
                    x3, y3 = float(parts[5]) * img_w, float(parts[6]) * img_h
                    x4, y4 = float(parts[7]) * img_w, float(parts[8]) * img_h

                    # 利用四边形的顶点逆向推导出原生属性
                    cx = (x1 + x2 + x3 + x4) / 4.0
                    cy = (y1 + y2 + y3 + y4) / 4.0
                    w = math.hypot(x2 - x1, y2 - y1)
                    h = math.hypot(x4 - x1, y4 - y1)
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                    # 重新生成带有完美手柄的 OBB 对象
                    shape = RotatedRectShape(cx, cy, w, h, angle, label)

                elif len(parts) > 9 and len(parts) % 2 == 1:
                    qpoints = [QPointF(float(parts[i]) * img_w, float(parts[i + 1]) * img_h) for i in
                               range(1, len(parts), 2)]
                    shape = PolyShape(QPolygonF(qpoints), label)
                else:
                    continue

                self._add_shape_to_scene(shape, label)
        except Exception as e:
            print(f"加载 YOLO 标注失败: {e}")

    def _load_xml(self, xml_path):
        import xml.etree.ElementTree as ET
        if not os.path.exists(xml_path): return
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name").text
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin, ymin = float(bndbox.find("xmin").text), float(bndbox.find("ymin").text)
                    xmax, ymax = float(bndbox.find("xmax").text), float(bndbox.find("ymax").text)
                    shape = RectShape(QRectF(xmin, ymin, xmax - xmin, ymax - ymin), label)
                    self._add_shape_to_scene(shape, label)
        except Exception as e:
            print(f"加载 XML 标注失败: {e}")

    def save_annotation(self, format_type):
        if not self.current_image_path or not self.scene.img_item:
            QMessageBox.warning(self, "提示", "请先打开图片")
            DialogOver(self, "请先在左侧树形目录中打开图片", "操作错误", "warning")
            return
        shapes_data = Exporter.extract_shapes(self.scene)
        # if not shapes_data:
        #     DialogOver(self, "当前画布没有标注内容可保存", "为空提示", "warning")
        #     return

        img_rect = self.scene.img_item.pixmap().rect()
        base_name = os.path.splitext(self.current_image_path)[0]

        try:
            if format_type == "json":
                out_path = base_name + ".json"
                Exporter.save_json(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)
            elif format_type == "yolo":
                out_path = base_name + ".txt"
                Exporter.save_yolo(out_path, img_rect.width(), img_rect.height(), shapes_data, self.class_list)
            elif format_type == "xml":
                out_path = base_name + ".xml"
                Exporter.save_xml(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)

            DialogOver(self, f"标注文件保存/更新成功！", "保存成功", "success")
            print(f"标注文件已保存到: {out_path}")
        except Exception as e:
            DialogOver(self, f"写入失败: {str(e)}", "保存出错", "danger")

    def closeEvent(self, event):
        self.auto_save_annotation()
        self.sam_client.cleanup()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # 撤销与重做快捷键拦截
        if key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            # Shift + Ctrl + Z 或者是多边形画点撤回处理
            if modifiers & Qt.ShiftModifier:
                self.redo()
            elif self.scene.mode == CanvasMode.POLY and len(self.scene.poly_pts) > 0:
                pass  # 多边形绘制中的撤销点由 canvas 自己处理，主窗口跳过
            else:
                self.undo()
        elif key == Qt.Key_Y and modifiers == Qt.ControlModifier:
            self.redo()

        if key == Qt.Key_D or key == Qt.Key_Right:
            current_idx = self.listFiles.currentRow()
            if current_idx < self.listFiles.count() - 1:
                self.listFiles.setCurrentRow(current_idx + 1)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            current_idx = self.listFiles.currentRow()
            if current_idx > 0:
                self.listFiles.setCurrentRow(current_idx - 1)
        elif key == Qt.Key_S and modifiers == Qt.ControlModifier:
            self.save_annotation(self.current_format)
        elif key == Qt.Key_E:  # 快捷键 E 修改类别
            selected_items = self.scene.selectedItems()
            for item in selected_items:
                if hasattr(item, 'label'):
                    self.edit_shape_label(item)
                    break

        elif key == Qt.Key_Q:
            if self.scene.mode == CanvasMode.POINT:
                DialogOver(self, "点标注模式下无法使用 SAM 智能提取", "提示", "warning")
            else:
                self.samSwitch.setChecked(not self.samSwitch.isChecked())
        elif key == Qt.Key_F1:
            self.show_help_dialog()
        elif key == Qt.Key_R:
            self.actionRect.trigger()
        elif key == Qt.Key_P:
            self.actionPoly.trigger()
        elif key == Qt.Key_T:
            self.actionPoint.trigger()
        elif key == Qt.Key_O:
            self.actionRBox.trigger()

        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())