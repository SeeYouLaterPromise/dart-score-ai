import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# 配置路径
input_img_dir = "/mnt/data/images_with_darts"         # 原始图像目录
input_label_dir = "/mnt/data/labels_with_dartpoints"  # 飞镖点标签（YOLO格式，多个点）
output_img_dir = "/mnt/data/dart_whole/images"        # 输出图像（飞镖crop后）
output_label_dir = "/mnt/data/dart_whole/labels"      # 输出标签（每张图一个 bbox）

# 创建输出目录
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 读取标签文件
img_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.jpg', '.png'))]
dart_size = 0.05  # 假设飞镖实际宽高为图像宽高的 5%

for img_file in tqdm(img_files, desc="Processing"):
    name, ext = os.path.splitext(img_file)
    img_path = os.path.join(input_img_dir, img_file)
    label_path = os.path.join(input_label_dir, name + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 读取 YOLO 格式标签（class cx cy w h）
    with open(label_path, 'r') as f:
        lines = f.readlines()

    dart_id = 0
    for line in lines[4:]:  # 前4个为参考点，后面为飞镖点
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, _, _ = map(float, parts)

        # 转为绝对坐标并裁切小图
        cx_abs, cy_abs = int(cx * w), int(cy * h)
        box_size = int(min(h, w) * dart_size)
        x1 = max(cx_abs - box_size, 0)
        y1 = max(cy_abs - box_size, 0)
        x2 = min(cx_abs + box_size, w)
        y2 = min(cy_abs + box_size, h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 保存裁切图像和标签（将整只飞镖标注为一个目标）
        dart_img_name = f"{name}_dart{dart_id}.jpg"
        dart_lbl_name = f"{name}_dart{dart_id}.txt"
        dart_id += 1

        cv2.imwrite(os.path.join(output_img_dir, dart_img_name), crop)

        # 生成中心为0.5的 YOLO bbox（因为整张图就是裁切后的飞镖图）
        with open(os.path.join(output_label_dir, dart_lbl_name), 'w') as out_f:
            out_f.write("0 0.5 0.5 1.0 1.0\n")

output_img_dir, output_label_dir
