# -*- coding: utf-8 -*-
"""
@Auth ：落花不写码
@File ：json_to_unet.py
@Motto :学习新思想，争做新青年
"""
import os
import json
import shutil
import random
import cv2
import numpy as np


def create_dir_structure(base_dir, use_test=False):
    dirs = ['train', 'val']
    if use_test:
        dirs.append('test')

    for split in dirs:
        os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'masks', split), exist_ok=True)


def generate_mask(json_path, class_mapping):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_h = data.get('imageHeight')
    img_w = data.get('imageWidth')

    # 创建全黑背景 (像素值为 0)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in class_mapping:
            continue

        class_id = class_mapping[label]
        shape_type = shape.get('shape_type', 'polygon')
        points = shape.get('points', [])

        if not points:
            continue

        pts = np.array(points, np.int32)

        # 处理矩形
        if shape_type == 'rectangle' and len(pts) == 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, thickness=-1)  # -1 表示实心填充

        # 处理多边形和旋转框
        elif shape_type in ['polygon', 'obb'] and len(pts) >= 3:
            cv2.fillPoly(mask, [pts], class_id)

        # 点标注，暂时用不到
        elif shape_type == 'point':
            cv2.circle(mask, (pts[0][0], pts[0][1]), radius=5, color=class_id, thickness=-1)

    return mask


def convert_to_unet_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    valid_pairs = []
    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(input_dir, filename)
            json_path = os.path.join(input_dir, f"{base_name}.json")

            if os.path.exists(json_path):
                valid_pairs.append((img_path, json_path))

    if not valid_pairs:
        print("未在输入目录中找到成对的 图片 和 json 文件！")
        return

    class_mapping = {"background": 0}
    classes_file = os.path.join(input_dir, 'classes.txt')

    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                cls_name = line.strip()
                if cls_name and cls_name not in class_mapping:
                    class_mapping[cls_name] = len(class_mapping)
    else:
        unique_classes = set()
        for _, json_path in valid_pairs:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    label = shape.get('label')
                    if label:
                        unique_classes.add(label)

        for cls_name in sorted(list(unique_classes)):
            class_mapping[cls_name] = len(class_mapping)

    print(f"加载的类别映射: {class_mapping}")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(input_dir, filename)
            json_path = os.path.join(input_dir, f"{base_name}.json")

            if os.path.exists(json_path):
                valid_pairs.append((img_path, json_path))

    if not valid_pairs:
        print("未在输入目录中找到成对的 图片 和 JSON 文件！")
        return

    print(f"找到 {len(valid_pairs)} 组有效数据，开始划分...")

    # 随机打乱并划分数据集
    random.shuffle(valid_pairs)
    total_count = len(valid_pairs)

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    # 输出目录
    use_test = test_ratio > 0 and len(test_pairs) > 0
    create_dir_structure(output_dir, use_test)

    def process_and_copy(pairs, split_name):
        print(f"正在处理 {split_name} 集 ({len(pairs)} 张)...")
        img_out_dir = os.path.join(output_dir, 'images', split_name)
        mask_out_dir = os.path.join(output_dir, 'masks', split_name)

        for img_path, json_path in pairs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # 拷贝原图
            shutil.copy(img_path, os.path.join(img_out_dir, os.path.basename(img_path)))

            # 生成并保存 Mask
            mask = generate_mask(json_path, class_mapping)
            mask_out_path = os.path.join(mask_out_dir, f"{base_name}.png")
            cv2.imencode('.png', mask)[1].tofile(mask_out_path)

    process_and_copy(train_pairs, 'train')
    process_and_copy(val_pairs, 'val')
    if use_test:
        process_and_copy(test_pairs, 'test')

    print(f"\n转换完成！数据集已保存至: {output_dir}")
    print(f"总计: 训练集={len(train_pairs)}, 验证集={len(val_pairs)}, 测试集={len(test_pairs)}")


if __name__ == '__main__':
    # 标注好的原图和 json 文件所在的目录
    input_dir = r"E:\11-AI\标注工具\LuoHuaLabel\yolotest"

    # U-Net 训练数据集要保存的输出目录
    output_dir = r"E:\11-AI\标注工具\LuoHuaLabel\unet_dataset"

    # 数据集划分比例,加起来等于1
    # 如果不需要测试集，可以设为: train_ratio=0.8, val_ratio=0.2, test_ratio=0
    train_ratio = 0.8
    val_ratio = 0.2
    test_ratio = 0

    convert_to_unet_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio)
