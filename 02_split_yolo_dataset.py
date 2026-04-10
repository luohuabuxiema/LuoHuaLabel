# -*- coding: utf-8 -*-
"""
@Auth ：落花不写码
@File ：split_yolo_dataset.py
@Motto :学习新思想，争做新青年
"""
import os
import random
import shutil


def create_yolo_dirs(output_dir, use_test=False):
    dirs = ['train', 'val']
    if use_test:
        dirs.append('test')

    for split in dirs:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)


def split_yolo_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        print("错误：train_ratio, val_ratio, test_ratio 之和必须等于 1")
        return

    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
    valid_pairs = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(input_dir, filename)
            txt_path = os.path.join(input_dir, f"{base_name}.txt")

            if os.path.exists(txt_path):
                valid_pairs.append((img_path, txt_path))

    if not valid_pairs:
        print("未在输入目录中找到成对的 图片 和 TXT 标签文件！请检查路径。")
        return

    print(f"共找到 {len(valid_pairs)} 组有效数据，开始随机打乱并划分...")

    # 随机打乱数据
    random.shuffle(valid_pairs)
    total_count = len(valid_pairs)

    # 计算各集合的数量
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    use_test = test_ratio > 0 and len(test_pairs) > 0
    create_yolo_dirs(output_dir, use_test)

    def copy_files(pairs, split_name):
        print(f"正在拷贝 {split_name} 集 ({len(pairs)} 张)...")
        img_out_dir = os.path.join(output_dir, 'images', split_name)
        txt_out_dir = os.path.join(output_dir, 'labels', split_name)

        for img_path, txt_path in pairs:
            shutil.copy(img_path, os.path.join(img_out_dir, os.path.basename(img_path)))
            shutil.copy(txt_path, os.path.join(txt_out_dir, os.path.basename(txt_path)))

    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')
    if use_test:
        copy_files(test_pairs, 'test')

    classes_file = os.path.join(input_dir, 'classes.txt')
    if os.path.exists(classes_file):
        shutil.copy(classes_file, os.path.join(output_dir, 'classes.txt'))

    print(f"\n数据集划分完成！已保存至: {output_dir}")
    print(f"数据分布: 训练集={len(train_pairs)}, 验证集={len(val_pairs)}, 测试集={len(test_pairs)}")


if __name__ == '__main__':
    # 原始的包含了图片和 .txt 标注文件的文件夹
    input_dir = r"E:\11-AI\标注工具\LuoHuaLabel\yolotest"

    # 划分后的 YOLO 数据集输出文件夹
    output_dir = r"E:\11-AI\标注工具\LuoHuaLabel\dataset"

    # 数据集划分比例,加起来等于1
    # 如果不需要测试集，可以设为: train_ratio=0.8, val_ratio=0.2, test_ratio=0
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
