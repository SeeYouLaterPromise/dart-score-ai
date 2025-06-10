import pandas as pd
import os
import shutil
import random
from tqdm import tqdm

DATA_DIR = os.path.join("..", "data")
# 参数设置
pkl_path = os.path.join(DATA_DIR, "darts_dataset", "labels.pkl")
image_root = os.path.join(DATA_DIR, "darts_dataset", "800")
output_root = os.path.join(DATA_DIR, "yolo_dataset")

train_ratio = 0.8
box_size = 0.05  # YOLO中目标边框大小（归一化宽高）

# 创建目录
for split in ['train', 'val']:
    os.makedirs(f"{output_root}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_root}/labels/{split}", exist_ok=True)

df = pd.read_pickle(pkl_path)
data = list(df.iterrows())
random.shuffle(data)

train_cut = int(len(data) * train_ratio)

for idx, (i, row) in tqdm(enumerate(data), total=len(data)):
    img_path = os.path.join(image_root, row["img_folder"], row["img_name"])
    if not os.path.exists(img_path):
        print(f"文件不存在，跳过：{img_path}")
        continue
    # print(img_path)
    split = 'train' if idx < train_cut else 'val'
    out_img_path = os.path.join(output_root, "images", split, f"{i}.jpg")
    out_lbl_path = os.path.join(output_root, "labels", split, f"{i}.txt")

    shutil.copy(img_path, out_img_path)

    with open(out_lbl_path, "w") as f:
        for j, (x, y) in enumerate(row["xy"]):
            cls_id = j if j < 4 else 4  # 0~3为参考点，4为飞镖
            f.write(f"{cls_id} {x:.6f} {y:.6f} {box_size} {box_size}\n")
