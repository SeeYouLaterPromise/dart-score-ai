import os
import shutil
from pathlib import Path
import random

def confirm_label_directory(label_dir):
    for txt_file in label_dir.rglob("*.txt"):
        return txt_file.parent.name  # 返回包含标签文件的目录名
    return None


def match_real_image_path(real_image_dir, dataset_name, line):
    image_name = line.strip().split('/')[-1]
    image_path = Path(real_image_dir) / dataset_name / image_name
    return image_path if image_path.exists() else None


def convert_and_split_yolo_datasets(
    source_folders,
    real_image_dir,
    output_base,
    train_ratio=0.8,
    category_names=None
):
    """
    合并多个 CVAT 导出的 Ultralytics YOLO Detection 1.0 数据并划分为 train/val 子集。

    参数：
        source_folders (list[str] or list[Path]): 每个文件夹都包含 train.txt、data.yaml、labels/
        output_base (str or Path): 输出根目录路径
        train_ratio (float): 训练集比例，默认 0.8
        num_classes (int): 类别数
        category_names (list[str]): 类别名列表，默认使用 [0, 1, ..., num_classes-1]

    输出：
        在 output_base 下生成 YOLOv8 所需格式的目录结构与 data.yaml 文件
    """
    num_classes = len(category_names) if category_names else 0
    output_base = Path(output_base)
    category_names = category_names or [str(i) for i in range(num_classes)]

    # 创建目录结构
    for subset in ['train', 'val']:
        (output_base / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (output_base / 'labels' / subset).mkdir(parents=True, exist_ok=True)

    all_items = []
    for folder in source_folders:
        folder = Path(folder)
        train_txt = folder / 'train.txt'
        # labels文件夹下面还有两层子文件夹train/dataset_name，在dataset_name下才有对应的标签文件，但是dataset_name不一定相同，应该recursive查找到含有标签文件的那一级文件夹才行，请你帮我完善。
        labels_dir = folder / 'labels'
        dataset_name = confirm_label_directory(labels_dir)
        labels_dir = labels_dir / "train" / dataset_name if dataset_name else "WrongDatasetName"

        with open(train_txt, 'r') as f:
            image_paths = [match_real_image_path(real_image_dir, dataset_name, line) for line in f.readlines() if line.strip()]

        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            label_path = labels_dir / img_path.with_suffix('.txt').name
            if img_path.exists() and label_path.exists():
                all_items.append((img_path, label_path))
            else:
                print(f"⚠️ 跳过缺失文件: {img_path} 或 {label_path}")

    # 划分 train/val
    random.shuffle(all_items)
    split_index = int(len(all_items) * train_ratio)
    train_items = all_items[:split_index]
    val_items = all_items[split_index:]

    def copy_items(items, subset):
        img_dir = output_base / 'images' / subset
        lbl_dir = output_base / 'labels' / subset
        for i, (img_path, label_path) in enumerate(items):
            new_img = img_dir / f'{subset}_{i}{img_path.suffix}'
            new_lbl = lbl_dir / f'{subset}_{i}.txt'
            shutil.copy(img_path, new_img)
            shutil.copy(label_path, new_lbl)

    copy_items(train_items, 'train')
    copy_items(val_items, 'val')

    # 写入 data.yaml
    with open(output_base / 'data.yaml', 'w') as f:
        f.write(f"""\
            path: {output_base.resolve()}
            train: images/train
            val: images/val
            nc: {num_classes}
            names: {category_names}
            """)

    print(f"✅ 处理完成！总计 {len(all_items)} 张图像，train: {len(train_items)}，val: {len(val_items)}")
    print(f"📁 输出目录：{output_base.resolve()}")

if __name__ == "__main__":
    source_dirs = [
        "../../job0",
        "../../job50",
        "../../job100",
        "../../job150",
    ]
    image_dir = "../../data/darts_dataset/800"
    output_dir = "../../data/yolo_digit"
    convert_and_split_yolo_datasets(
        source_folders=source_dirs,
        real_image_dir=image_dir,
        output_base=output_dir,
        train_ratio=0.8,
        category_names=["0", "1", "2", "3", "4", "5", "6", "7"]
    )
