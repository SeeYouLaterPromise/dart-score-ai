import os
import shutil
import random
import h5py
import cv2
from tqdm import tqdm

# DATA_DIR = os.path.join("..", "data")
DATA_DIR = "data"

mat_path = os.path.join(DATA_DIR, "SVHN", "digitStruct.mat")
image_root = os.path.join(DATA_DIR, "SVHN")  # 图像目录
output_root = os.path.join(DATA_DIR, "yolo_dataset_2")  # 输出目录

train_ratio = 0.8

# 创建目录结构
for split in ['train', 'val']:
    os.makedirs(f"{output_root}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_root}/labels/{split}", exist_ok=True)

# ========== digitStruct.mat 读取函数 ==========
def get_image_name(ds, idx):
    name_ref = ds['digitStruct']['name'][idx][0]
    return ''.join([chr(ds[name_ref][i][0]) for i in range(len(ds[name_ref]))])

def get_bbox(ds, idx):
    bbox_ref = ds['digitStruct']['bbox'][idx].item()
    def _get(attr):
        field = ds[bbox_ref][attr]
        if field.shape[0] == 1:
            return [field[0][0]]
        else:
            return [ds[field[i][0]][()][0][0] for i in range(field.shape[0])]
    return {
        'label': _get('label'),
        'left': _get('left'),
        'top': _get('top'),
        'width': _get('width'),
        'height': _get('height'),
    }

# ========== 主转换逻辑 ==========
with h5py.File(mat_path, 'r') as ds:
    total = ds['digitStruct']['name'].shape[0]
    indices = list(range(total))
    random.shuffle(indices)
    train_cut = int(total * train_ratio)

    for idx, i in tqdm(enumerate(indices), total=total):
        split = 'train' if idx < train_cut else 'val'

        name = get_image_name(ds, i)
        bbox = get_bbox(ds, i)
        src_img_path = os.path.join(image_root, name)
        dst_img_path = os.path.join(output_root, "images", split, f"{i}.jpg")
        dst_lbl_path = os.path.join(output_root, "labels", split, f"{i}.txt")

        # 图像存在性与尺寸读取
        if not os.path.exists(src_img_path):
            print(f"❌ 文件不存在：{src_img_path}")
            continue
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"❌ 图像无法读取：{src_img_path}")
            continue
        h, w = img.shape[:2]

        # 写入图像
        shutil.copy(src_img_path, dst_img_path)

        with open(dst_lbl_path, "w") as f:
            for j in range(len(bbox["label"])):
                x = bbox["left"][j]
                y = bbox["top"][j]
                bw = bbox["width"][j]
                bh = bbox["height"][j]
                label = int(bbox["label"][j]) % 10  # 10 表示数字 0

                # 坐标归一化
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                # YOLO 标签格式：class_id x_center y_center width height
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

print("\n✅ SVHN .mat → YOLO 标签转换完成！")
