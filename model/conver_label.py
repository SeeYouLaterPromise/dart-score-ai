import pandas as pd
import os

IMG_SIZE = 800
BOX_SIZE = 0.05  # YOLO中宽高都设为5%（40px）

label_df = pd.read_pickle("data/labels.pkl")
output_dir = "data/yolo_labels/"
os.makedirs(output_dir, exist_ok=True)

for idx, row in label_df.iterrows():
    img_name = os.path.splitext(row["img_name"])[0]  # 去掉.jpg
    label_path = os.path.join(output_dir, f"{img_name}.txt")

    with open(label_path, "w") as f:
        for i, (x, y) in enumerate(row["xy"]):
            cls_id = i if i < 4 else 4  # 参考点0~3，飞镖是4类
            f.write(f"{cls_id} {x:.6f} {y:.6f} {BOX_SIZE} {BOX_SIZE}\n")
