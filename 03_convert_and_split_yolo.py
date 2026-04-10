# -*- coding: utf-8 -*-
"""
@Auth ：落花不写码
@File ：03_convert_and_split_yolo.py
@Motto :学习新思想，争做新青年
"""
import os
import json
import random
import shutil
import cv2
import xml.etree.ElementTree as ET

import numpy as np


def create_yolo_dirs(output_dir, use_test=False):
    dirs = ['train', 'val']
    if use_test:
        dirs.append('test')

    for split in dirs:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)


def parse_json_to_yolo(json_path, img_w, img_h, class_map):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 优先使用 json 内的宽高
    img_w = data.get('imageWidth', img_w)
    img_h = data.get('imageHeight', img_h)

    if img_w <= 0 or img_h <= 0:
        return []

    lines = []
    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in class_map:
            continue

        cls_id = class_map[label]
        shape_type = shape.get('shape_type', 'polygon')
        points = shape.get('points', [])

        if not points: continue

        # 矩形 (Rectangle) -> 标准 bbox
        if shape_type == 'rectangle' and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 多边形 (Polygon) 或 旋转框 (OBB) -> 实例分割多点轮廓
        elif shape_type in ['polygon', 'obb']:
            pts_normalized = []
            for pt in points:
                pts_normalized.extend([f"{pt[0] / img_w:.6f}", f"{pt[1] / img_h:.6f}"])
            lines.append(f"{cls_id} " + " ".join(pts_normalized))

        # todo： 点标注 (Point) -> 待测试
        elif shape_type == 'point' and len(points) == 1:
            cx = points[0][0] / img_w
            cy = points[0][1] / img_h
            w = 0.02
            h = 0.02
            cx = max(w / 2, min(1.0 - w / 2, cx))
            cy = max(h / 2, min(1.0 - h / 2, cy))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def parse_xml_to_yolo(xml_path, img_w, img_h, class_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is not None:
        try:
            img_w = float(size.find('width').text)
            img_h = float(size.find('height').text)
        except:
            pass

    if img_w <= 0 or img_h <= 0:
        return []

    lines = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in class_map:
            continue
        cls_id = class_map[label]

        bndbox = obj.find('bndbox')
        polygon = obj.find('polygon')
        robndbox = obj.find('robndbox')

        if bndbox is not None:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            cx = (xmin + xmax) / 2.0 / img_w
            cy = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 多边形
        elif polygon is not None:
            pts_normalized = []
            for pt in polygon.findall('pt'):
                x = float(pt.find('x').text) / img_w
                y = float(pt.find('y').text) / img_h
                pts_normalized.extend([f"{x:.6f}", f"{y:.6f}"])
            if pts_normalized:
                lines.append(f"{cls_id} " + " ".join(pts_normalized))

        # OBB 旋转框
        elif robndbox is not None:
            cx = float(robndbox.find('cx').text) / img_w
            cy = float(robndbox.find('cy').text) / img_h
            w = float(robndbox.find('w').text) / img_w
            h = float(robndbox.find('h').text) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def build_class_mapping(input_dir, valid_pairs):
    class_mapping = {}
    classes_file = os.path.join(input_dir, 'classes.txt')

    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                cls_name = line.strip()
                if cls_name and cls_name not in class_mapping:
                    class_mapping[cls_name] = len(class_mapping)
    else:
        print("未找到 classes.txt，正在从 json/xml 中提取所有类别...")
        unique_classes = set()
        for _, label_path, label_type in valid_pairs:
            if label_type == 'json':
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for shape in data.get('shapes', []):
                        if shape.get('label'):
                            unique_classes.add(shape['label'])
            elif label_type == 'xml':
                tree = ET.parse(label_path)
                for obj in tree.getroot().findall('object'):
                    if obj.find('name') is not None:
                        unique_classes.add(obj.find('name').text)

        for cls_name in sorted(list(unique_classes)):
            class_mapping[cls_name] = len(class_mapping)

    return class_mapping


def split_yolo_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        print("错误：train_ratio, val_ratio, test_ratio 比例之和必须等于 1")
        return

    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
    valid_pairs = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(input_dir, filename)

            json_path = os.path.join(input_dir, f"{base_name}.json")
            xml_path = os.path.join(input_dir, f"{base_name}.xml")

            if os.path.exists(json_path):
                valid_pairs.append((img_path, json_path, 'json'))
            elif os.path.exists(xml_path):
                valid_pairs.append((img_path, xml_path, 'xml'))

    if not valid_pairs:
        print("未在输入目录中找到任何对应的文件")
        return

    class_mapping = build_class_mapping(input_dir, valid_pairs)
    print(f"使用的类别映射: {class_mapping}")

    # 随机打乱并计算分割数量
    random.shuffle(valid_pairs)
    total_count = len(valid_pairs)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    splits = {
        'train': valid_pairs[:train_count],
        'val': valid_pairs[train_count:train_count + val_count],
    }
    use_test = test_ratio > 0 and (train_count + val_count < total_count)
    if use_test:
        splits['test'] = valid_pairs[train_count + val_count:]

    create_yolo_dirs(output_dir, use_test)

    for split_name, pairs in splits.items():
        print(f"正在处理 {split_name} 集 ({len(pairs)} 张图片)...")
        img_out_dir = os.path.join(output_dir, 'images', split_name)
        txt_out_dir = os.path.join(output_dir, 'labels', split_name)

        for img_path, label_path, label_type in pairs:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # 拷贝图片
            out_img_path = os.path.join(img_out_dir, os.path.basename(img_path))
            shutil.copy(img_path, out_img_path)

            # 解析为 YOLO 格式文本
            if label_type == 'json':
                yolo_lines = parse_json_to_yolo(label_path, img_w, img_h, class_mapping)
            else:
                yolo_lines = parse_xml_to_yolo(label_path, img_w, img_h, class_mapping)

            # 写入 TXT 文件
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_txt_path = os.path.join(txt_out_dir, f"{base_name}.txt")
            with open(out_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

    out_classes_file = os.path.join(output_dir, 'classes.txt')
    with open(out_classes_file, 'w', encoding='utf-8') as f:
        sorted_classes = sorted(class_mapping.keys(), key=lambda k: class_mapping[k])
        for c in sorted_classes:
            f.write(f"{c}\n")

    print(f"\n数据集转换与划分完毕！输出目录: {output_dir}")
    print(f"数据分布: 训练集={len(splits['train'])}, 验证集={len(splits['val'])}, 测试集={len(splits.get('test', []))}")


if __name__ == '__main__':
    # 原始包含了图片和标注文件 (json 或 xml) 的文件夹
    input_dir = r"E:\11-AI\标注工具\LuoHuaLabel\yolotest"

    # 划分后的 YOLO 数据集输出文件夹
    output_dir = r"E:\11-AI\标注工具\LuoHuaLabel\dataset"

    # 数据集划分比例, 加起来等于1
    train_ratio = 0.8
    val_ratio = 0.2
    test_ratio = 0

    split_yolo_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio)


# ==============================运行完，数据集格式结构如下==============================
# dataset_xxx/
# ├── images/          # 存放所有图片
# │   ├── train/
# │   ├── val/
# │   └── test/        # (如果 test_ratio > 0 才会生成)
# └── labels/          # 存放所有对应的 txt 标签文件
#     ├── train/
#     ├── val/
#     └── test/
