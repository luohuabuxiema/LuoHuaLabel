# -*- coding: utf-8 -*-
"""
@Auth ：落花不写码
@File ：main_dataset_tool.py
@Motto :学习新思想，争做新青年
"""
import sys
import os
import random
import shutil
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
                               QTextEdit, QDialog, QSpinBox, QMessageBox, QFrame, QGridLayout,
                               QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QTextCursor, QColor

from utils.message import DialogOver

from PySide6.QtWidgets import QProxyStyle, QStyle
from PySide6.QtGui import QPolygon, QPainter, QColor
from PySide6.QtCore import QPoint


class DatasetWorker(QThread):
    log_signal = Signal(str)
    finish_signal = Signal(bool, str)  # 完成信号

    def __init__(self, mode, img_dir, ann_dir, out_dir, ratios):
        super().__init__()
        self.mode = mode
        self.img_dir = img_dir
        self.ann_dir = ann_dir if ann_dir else img_dir
        self.out_dir = out_dir
        self.train_r, self.val_r, self.test_r = ratios

    def log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        try:
            self.log(f"🚀 开始任务...")
            self.log(f"📁 图片目录: {self.img_dir}")
            self.log(f"📁 标签目录: {self.ann_dir}")
            self.log(f"📁 输出目录: {self.out_dir}")
            self.log(f"📊 划分比例: Train({self.train_r}), Val({self.val_r}), Test({self.test_r})")
            self.log("-" * 50)

            if self.mode == "UNET":
                self.process_unet()
            elif self.mode == "YOLO_SPLIT":
                self.process_yolo_split()
            elif self.mode == "TO_YOLO":
                self.process_convert_to_yolo()

        except Exception as e:
            self.log(f"❌ 错误: {str(e)}")
            self.finish_signal.emit(False, str(e))

    def get_valid_pairs(self, exts, label_exts, use_json_xml_mix=False):
        valid_pairs = []
        for filename in os.listdir(self.img_dir):
            if filename.lower().endswith(exts):
                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.img_dir, filename)

                if use_json_xml_mix:
                    json_path = os.path.join(self.ann_dir, f"{base_name}.json")
                    xml_path = os.path.join(self.ann_dir, f"{base_name}.xml")
                    if os.path.exists(json_path):
                        valid_pairs.append((img_path, json_path, 'json'))
                    elif os.path.exists(xml_path):
                        valid_pairs.append((img_path, xml_path, 'xml'))
                else:
                    for l_ext in label_exts:
                        l_path = os.path.join(self.ann_dir, f"{base_name}{l_ext}")
                        if os.path.exists(l_path):
                            valid_pairs.append((img_path, l_path))
                            break
        return valid_pairs

    def split_data(self, valid_pairs):
        random.shuffle(valid_pairs)
        total = len(valid_pairs)
        train_c = int(total * self.train_r)
        val_c = int(total * self.val_r)

        train_p = valid_pairs[:train_c]
        val_p = valid_pairs[train_c:train_c + val_c]
        test_p = valid_pairs[train_c + val_c:]
        return train_p, val_p, test_p

    # ------------------  转 U-Net ------------------
    def process_unet(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), ('.json',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON！")
            return

        self.log(f"✅ 找到 {len(valid_pairs)} 组有效数据。")
        class_mapping = {"background": 0}
        self.log("正在从 JSON 提取类别...")
        unique_classes = set()
        for _, j_path in valid_pairs:
            with open(j_path, 'r', encoding='utf-8') as f:
                for s in json.load(f).get('shapes', []):
                    if s.get('label'): unique_classes.add(s['label'])
        for c in sorted(list(unique_classes)):
            class_mapping[c] = len(class_mapping)
        self.log(f"🏷️ 类别映射: {class_mapping}")

        train_p, val_p, test_p = self.split_data(valid_pairs)

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'masks', split_name), exist_ok=True)
            self.log(f"📦 正在生成 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, json_p in pairs:
                base_name = os.path.splitext(os.path.basename(img_p))[0]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                with open(json_p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mask = np.zeros((data.get('imageHeight', 100), data.get('imageWidth', 100)), dtype=np.uint8)
                for shape in data.get('shapes', []):
                    cid = class_mapping.get(shape.get('label'))
                    if cid is None: continue
                    pts = np.array(shape.get('points', []), np.int32)
                    if shape.get('shape_type') == 'rectangle' and len(pts) == 2:
                        cv2.rectangle(mask, pts[0], pts[1], cid, -1)
                    elif len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], cid)
                    elif shape.get('shape_type') == 'point':
                        cv2.circle(mask, pts[0], 5, cid, -1)

                cv2.imencode('.png', mask)[1].tofile(
                    os.path.join(self.out_dir, 'masks', split_name, f"{base_name}.png"))

        self.log("\n🎉 U-Net 格式数据集生成完毕！")
        self.finish_signal.emit(True, "U-Net 数据集处理成功")

    # ------------------ 生成 data.yaml ------------------
    def generate_yaml(self, class_map):
        yaml_path = os.path.join(self.out_dir, "data.yaml")
        abs_out_dir = os.path.abspath(self.out_dir).replace("\\", "/")

        train_path = f"{abs_out_dir}/images/train"
        val_path = f"{abs_out_dir}/images/val"
        test_path = f"{abs_out_dir}/images/test"

        nc = len(class_map)
        names_dict = {v: k for k, v in class_map.items()}
        yaml_content = []
        yaml_content.append(f"# Train and Val data path")
        yaml_content.append(f"train: {train_path}")
        yaml_content.append(f"val: {val_path}")
        if self.test_r > 0:
            yaml_content.append(f"test: {test_path}")

        yaml_content.append(f"\n# Number of classes")
        yaml_content.append(f"nc: {nc}")

        yaml_content.append(f"\n# Class names")
        yaml_content.append(f"names:")
        for i in range(nc):
            yaml_content.append(f"  {i}: {names_dict[i]}")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yaml_content))

        self.log(f"📝 已自动生成配置文件: {yaml_path}")

    # ------------------ YOLO 纯划分 ------------------
    def process_yolo_split(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), ('.txt',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和TXT！")
            return

        class_map = {}
        classes_txt = os.path.join(self.ann_dir, "classes.txt")
        if os.path.exists(classes_txt):
            with open(classes_txt, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip(): class_map[line.strip()] = i
        else:
            class_map = {"class_0": 0}

        self.log(f"✅ 找到 {len(valid_pairs)} 组有效数据。")
        train_p, val_p, test_p = self.split_data(valid_pairs)

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"📦 正在拷贝 {split_name} 集 ({len(pairs)} 张)...")
            for img_p, txt_p in pairs:
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))
                shutil.copy(txt_p, os.path.join(self.out_dir, 'labels', split_name, os.path.basename(txt_p)))

        self.generate_yaml(class_map)
        self.log("\n🎉 YOLO 数据集划分完毕！")
        self.finish_signal.emit(True, "YOLO 划分处理成功")

    # ------------------ XML/JSON 转 YOLO ------------------
    def process_convert_to_yolo(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), [], use_json_xml_mix=True)
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON/XML！")
            return

        self.log(f"✅ 找到 {len(valid_pairs)} 组有效数据，正在提取类别...")
        class_map = {}
        unique_classes = set()
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    for s in json.load(f).get('shapes', []):
                        if s.get('label'): unique_classes.add(s['label'])
            elif l_type == 'xml':
                for obj in ET.parse(l_path).getroot().findall('object'):
                    if obj.find('name') is not None: unique_classes.add(obj.find('name').text)
        for c in sorted(list(unique_classes)):
            class_map[c] = len(class_map)
        self.log(f"🏷️ 类别映射: {class_map}")

        train_p, val_p, test_p = self.split_data(valid_pairs)
        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"📦 处理 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, label_p, l_type in pairs:
                # 读取图片宽高
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]

                # 拷贝图片
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                lines = []

                # ================= JSON 格式解析 =================
                if l_type == 'json':
                    with open(label_p, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            cid = class_map.get(s.get('label'))
                            if cid is None or not s.get('points'): continue

                            pts = s['points']
                            shape_type = s.get('shape_type', 'polygon')

                            # 常规矩形 (HBB)
                            if shape_type == 'rectangle' and len(pts) == 2:
                                cx, cy = (pts[0][0] + pts[1][0]) / 2 / w, (pts[0][1] + pts[1][1]) / 2 / h
                                bw, bh = abs(pts[1][0] - pts[0][0]) / w, abs(pts[1][1] - pts[0][1]) / h
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                            #  旋转框 (OBB)
                            elif shape_type == 'obb':
                                obb_pts = pts[:4]
                                pts_normalized = []
                                for pt in obb_pts:
                                    pts_normalized.extend([f"{pt[0] / w:.6f}", f"{pt[1] / h:.6f}"])
                                lines.append(f"{cid} " + " ".join(pts_normalized))

                            # 多边形实例分割 (Polygon)
                            elif shape_type == 'polygon':
                                pts_normalized = []
                                for pt in pts:
                                    pts_normalized.extend([f"{pt[0] / w:.6f}", f"{pt[1] / h:.6f}"])
                                lines.append(f"{cid} " + " ".join(pts_normalized))

                            # 点标注 (Point) - 转换为中心微小框
                            elif shape_type == 'point' and len(pts) == 1:
                                cx, cy = pts[0][0] / w, pts[0][1] / h
                                pw, ph = 0.02, 0.02  # 设为全图宽高的 2%
                                cx = max(pw / 2, min(1.0 - pw / 2, cx))
                                cy = max(ph / 2, min(1.0 - ph / 2, cy))
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}")

                # ================= XML 格式解析 (补全) =================
                elif l_type == 'xml':
                    root = ET.parse(label_p).getroot()
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        cid = class_map.get(label)
                        if cid is None: continue

                        bndbox = obj.find('bndbox')
                        polygon = obj.find('polygon')
                        robndbox = obj.find('robndbox')

                        # 标准矩形
                        if bndbox is not None:
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            cx, cy = (xmin + xmax) / 2.0 / w, (ymin + ymax) / 2.0 / h
                            bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                        # 多边形
                        elif polygon is not None:
                            pts_normalized = []
                            for pt in polygon.findall('pt'):
                                px = float(pt.find('x').text) / w
                                py = float(pt.find('y').text) / h
                                pts_normalized.extend([f"{px:.6f}", f"{py:.6f}"])
                            if pts_normalized:
                                lines.append(f"{cid} " + " ".join(pts_normalized))

                        # 旋转框
                        elif robndbox is not None:
                            cx = float(robndbox.find('cx').text) / w
                            cy = float(robndbox.find('cy').text) / h
                            bw = float(robndbox.find('w').text) / w
                            bh = float(robndbox.find('h').text) / h
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # 写入 TXT 文件
                txt_out = os.path.join(self.out_dir, 'labels', split_name,
                                       os.path.splitext(os.path.basename(img_p))[0] + ".txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        # 写入 classes.txt
        with open(os.path.join(self.out_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in sorted(class_map.keys(), key=lambda k: class_map[k]): f.write(f"{c}\n")

        self.generate_yaml(class_map)
        self.log("\n🎉 JSON/XML 转 YOLO 完成！")
        self.finish_signal.emit(True, "格式转换与划分成功")


class RatioDialog(QDialog):
    def __init__(self, current_ratios, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置划分比例")
        self.setFixedSize(340, 240)
        self.setStyleSheet("""
            QDialog { background-color: #ffffff; }
            QLabel { font-family: "Microsoft YaHei"; font-size: 13px; color: #303133; }
            QSpinBox {
                border: 1px solid #dcdfe6;
                border-radius: 4px;
                padding: 6px;
                font-size: 13px;
                color: #606266;
            }
            QSpinBox:focus { border: 1px solid #409eff; }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                background-color: #f5f7fa;
                border-left: 1px solid #dcdfe6;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #e9e9eb; }
            
            QPushButton {
                background-color: #409eff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #66b1ff; }
            QPushButton:pressed { background-color: #3a8ee6; }
        """)
        self.ratios = current_ratios

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header_label = QLabel("请分配数据集百分比（总和必须为 100%）")
        header_label.setStyleSheet("color: #909399; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(header_label)

        self.train_box = self.create_row("训练集 (Train) :", current_ratios[0] * 100, layout)
        self.val_box = self.create_row("验证集 (Val) :", current_ratios[1] * 100, layout)
        self.test_box = self.create_row("测试集 (Test) :", current_ratios[2] * 100, layout)

        layout.addStretch()
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("保存")
        self.btn_ok.clicked.connect(self.check_and_accept)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def create_row(self, text, val, layout):
        row = QHBoxLayout()
        label = QLabel(text)
        label.setFixedWidth(120)
        row.addWidget(label)

        box = QSpinBox()
        box.setRange(0, 100)
        box.setValue(int(val))
        box.setSuffix(" %")
        row.addWidget(box)
        layout.addLayout(row)
        return box

    def check_and_accept(self):
        t, v, te = self.train_box.value(), self.val_box.value(), self.test_box.value()
        if t + v + te != 100:
            QMessageBox.warning(self, "错误", "三个比例之和必须等于 100%！")
            return
        self.ratios = (t / 100.0, v / 100.0, te / 100.0)
        self.accept()


class DatasetToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LuoHuaLabel - 数据集处理系统")
        self.resize(850, 650)

        # 整体应用偏灰色的背景，凸显内部白色卡片
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f2f5; }
            QWidget { font-family: "Microsoft YaHei", sans-serif; }

            QLabel { font-size: 13px; color: #606266; font-weight: bold; }

            /* 表单输入框美化 */
            QLineEdit { 
                border: 1px solid #dcdfe6; 
                border-radius: 4px; 
                padding: 8px 12px; 
                background: white; 
                font-size: 13px; 
                color: #303133;
            }
            QLineEdit:focus { border: 1px solid #409eff; }

            /* 默认按钮美化 */
            QPushButton#DefaultBtn { 
                border-radius: 4px; 
                padding: 8px 15px; 
                font-weight: bold; 
                font-size: 13px; 
                border: 1px solid #dcdfe6; 
                background: white; 
                color: #606266; 
            }
            QPushButton#DefaultBtn:hover { color: #409eff; border-color: #c6e2ff; background-color: #ecf5ff; }

            /* 下拉框美化 */
            QComboBox { 
                border: 1px solid #dcdfe6; 
                border-radius: 4px; 
                padding: 8px 12px; 
                background: white; 
                color: #606266;
            }
            QComboBox:hover { border: 1px solid #c0c4cc; }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox QAbstractItemView {
                border: 1px solid #e4e7ed;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #f5f7fa;
                selection-color: #409eff;
            }
            
            /* 控制台终端美化 */
            QTextEdit { 
                background-color: #1e1e1e; 
                color: #67c23a; 
                font-family: Consolas, monospace; 
                border-radius: 6px; 
                padding: 12px; 
                font-size: 13px;
                border: 1px solid #333333;
            }
        """)

        self.ratios = (0.8, 0.2, 0.0)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # -------------------------------------------------------------
        # 包含配置和表单
        # -------------------------------------------------------------
        card_frame = QFrame()
        card_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #ebeef5;
            }
        """)
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 15))
        shadow.setOffset(0, 2)
        card_frame.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(20, 25, 20, 25)
        card_layout.setSpacing(15)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(12)
        grid_layout.setColumnStretch(1, 1)

        self.img_input = QLineEdit()
        self.img_input.setPlaceholderText("请选择/填写包含原图的文件夹路径...")
        btn_img = QPushButton("选择目录")
        btn_img.setObjectName("DefaultBtn")
        btn_img.clicked.connect(lambda: self.select_dir(self.img_input))
        grid_layout.addWidget(QLabel("🖼️ 图片数据目录:"), 0, 0)
        grid_layout.addWidget(self.img_input, 0, 1)
        grid_layout.addWidget(btn_img, 0, 2)

        self.ann_input = QLineEdit()
        self.ann_input.setPlaceholderText("（可选）为空则代表标签和图片在同一个目录中")
        btn_ann = QPushButton("选择目录")
        btn_ann.setObjectName("DefaultBtn")
        btn_ann.clicked.connect(lambda: self.select_dir(self.ann_input))
        grid_layout.addWidget(QLabel("📑 标签数据目录:"), 1, 0)
        grid_layout.addWidget(self.ann_input, 1, 1)
        grid_layout.addWidget(btn_ann, 1, 2)

        self.out_input = QLineEdit()
        self.out_input.setPlaceholderText("请选择/填写处理后数据集的存放目录...")
        btn_out = QPushButton("选择目录")
        btn_out.setObjectName("DefaultBtn")
        btn_out.clicked.connect(lambda: self.select_dir(self.out_input))
        grid_layout.addWidget(QLabel("📦 最终输出目录:"), 2, 0)
        grid_layout.addWidget(self.out_input, 2, 1)
        grid_layout.addWidget(btn_out, 2, 2)

        card_layout.addLayout(grid_layout)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border-top: 1px solid #ebeef5; margin: 5px 0px;")
        card_layout.addWidget(line)

        # 模式与比例配置行
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("⚙️ 处理模式配置:"))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "模式一: 纯划分 YOLO (TXT) 打乱划分",
            "模式二: 转换 JSON/XML -> YOLO格式并划分",
            "模式三: 转换 JSON -> U-Net (Mask掩码)",
        ])
        self.mode_combo.setMinimumWidth(300)
        opt_layout.addWidget(self.mode_combo)
        opt_layout.addStretch()

        self.ratio_btn = QPushButton("⚖️ 比例: 80% Train | 20% Val")
        self.ratio_btn.setObjectName("DefaultBtn")
        self.ratio_btn.clicked.connect(self.open_ratio_dialog)
        opt_layout.addWidget(self.ratio_btn)

        card_layout.addLayout(opt_layout)
        main_layout.addWidget(card_frame)

        # -------------------------------------------------------------
        # 第二部分：执行按钮
        # -------------------------------------------------------------
        self.start_btn = QPushButton("🚀 开始执行转换与划分")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton { 
                background-color: #409eff; 
                color: white; 
                font-size: 16px; 
                font-weight: bold;
                padding: 12px; 
                border-radius: 6px; 
                border: none; 
            }
            QPushButton:hover { background-color: #66b1ff; }
            QPushButton:pressed { background-color: #3a8ee6; }
            QPushButton:disabled { background-color: #a0cfff; color: #f0f0f0; }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_btn)

        # -------------------------------------------------------------
        # 运行控制台
        # -------------------------------------------------------------
        console_label = QLabel("🖥️ 处理日志台")
        console_label.setStyleSheet("color: #909399; font-size: 12px; margin-top: 5px;")
        main_layout.addWidget(console_label)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        # 让控制台占据下方所有的剩余空间
        main_layout.addWidget(self.console, 1)

    def select_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d: line_edit.setText(d)

    def open_ratio_dialog(self):
        dlg = RatioDialog(self.ratios, self)
        if dlg.exec():
            self.ratios = dlg.ratios
            t, v, te = [int(r * 100) for r in self.ratios]
            txt = f"⚖️ 比例: {t}% Train | {v}% Val"
            if te > 0:
                txt += f" | {te}% Test"
            self.ratio_btn.setText(txt)

    def append_log(self, text):
        self.console.append(text)
        self.console.moveCursor(QTextCursor.End)

    def trigger_message(self, text, title, status):
        if DialogOver:
            DialogOver(self, text, title, status)
        else:
            QMessageBox.information(self, title, text)

    def start_processing(self):
        img_dir = self.img_input.text().strip()
        out_dir = self.out_input.text().strip()
        ann_dir = self.ann_input.text().strip()

        if not img_dir or not out_dir:
            self.trigger_message("请先选择源图片目录和输出目录！", "表单未完成", "warning")
            return

        mode_idx = self.mode_combo.currentIndex()
        mode_map = {0: "YOLO_SPLIT", 1: "TO_YOLO", 2: "UNET"}

        self.console.clear()
        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳ 后台处理中，请稍候...")

        self.worker = DatasetWorker(mode_map[mode_idx], img_dir, ann_dir, out_dir, self.ratios)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finish_signal.connect(self.on_process_finish)
        self.worker.start()

    def on_process_finish(self, success, msg):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("🚀 开始执行转换与划分")
        if success:
            self.trigger_message(msg, "处理完成", "success")
        else:
            self.trigger_message(msg, "处理异常", "danger")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetToolWindow()
    window.show()
    sys.exit(app.exec())