import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# 输入多个任务数据的根目录，例如每个子文件夹里包含 train.txt 和 labels/
multi_task_root = 'multi_tasks'
# 输出统一结构的数据集目录
output_root = 'data/darts-digit'
image_output_dir = os.path.join(output_root, 'images')
label_output_dir = os.path.join(output_root, 'labels')
os.makedirs(image_output_dir + '/train', exist_ok=True)
os.makedirs(image_output_dir + '/val', exist_ok=True)
os.makedirs(label_output_dir + '/train', exist_ok=True)
os.makedirs(label_output_dir + '/val', exist_ok=True)

# 初始化新 data.yaml 的类别列表
all_classes = set()

# 用于给图像和标签重命名防止冲突
global_idx = 0

# 遍历每个子任务
for task_dir in os.listdir(multi_task_root):
    task_path = os.path.join(multi_task_root, task_dir)
    if not os.path.isdir(task_path):
        continue

    train_txt = os.path.join(task_path, 'train.txt')
    labels_dir = os.path.join(task_path, 'labels')
    yaml_path = os.path.join(task_path, 'data.yaml')

    if not os.path.exists(train_txt) or not os.path.exists(labels_dir):
        print(f"跳过 {task_dir}，缺少必要文件。")
        continue

    # 收集类别信息
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        names = data_yaml.get('names', [])
        all_classes.update(names)

    with open(train_txt, 'r') as f:
        lines = f.read().splitlines()

    for img_path in tqdm(lines, desc=f"Processing {task_dir}"):
        img_path = img_path.strip()
        if not os.path.exists(img_path):
            print(f"图像不存在：{img_path}")
            continue

        label_path = os.path.join(labels_dir, Path(img_path).stem + '.txt')
        if not os.path.exists(label_path):
            print(f"标签不存在：{label_path}")
            continue

        # 简单划分 train/val：5%作为val
        split = 'val' if global_idx % 20 == 0 else 'train'

        # 统一命名避免冲突
        new_name = f"{task_dir}_{global_idx:06d}"
        img_dst = os.path.join(image_output_dir, split, new_name + Path(img_path).suffix)
        label_dst = os.path.join(label_output_dir, split, new_name + '.txt')

        shutil.copy2(img_path, img_dst)
        shutil.copy2(label_path, label_dst)

        global_idx += 1

# 写入新的 data.yaml
class_list = sorted(list(all_classes))
new_yaml = {
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(class_list),
    'names': class_list
}
with open(os.path.join(output_root, 'data.yaml'), 'w') as f:
    yaml.dump(new_yaml, f, allow_unicode=True)

print("✅ 数据整合完成，已生成统一格式的数据集。")
