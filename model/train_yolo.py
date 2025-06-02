from yolov5 import train
import os
from pathlib import Path
from ultralytics import YOLO

os.environ["YOLO_OVERWRITE_CACHE"] = "1"    # 强制覆盖缓存
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # solve: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]

weight_path = "./yolov5s.pt"

data_path = PROJECT_ROOT / "data" / "yolo_digit" / "data.yaml"
project_path = PROJECT_ROOT / "runs_digit"
hyp_path = PROJECT_ROOT / "config/digit_hyp.yaml"
cfg_path = PROJECT_ROOT / "config" / "yolov5_dart_digit.yaml"

def train_yolo5(log_name="yolov5_200_third"):
    train.run(
        img=640,
        batch=4,
        epochs=150,
        data=str(data_path),
        weights=str(weight_path),
        hyp=hyp_path,
        cfg=cfg_path,
        project=str(project_path),
        name=log_name,
        patience=50,  # patience 指的是你愿意容忍验证指标“连续不提升”的 epoch 数量。
    )

def train_yolo8(log_name="yolov8_200"):
    # load the model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data=str(data_path),

        batch=16,
        epochs=80,
        optimizer="Adam",
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=20,

        # 数据增强参数
        hsv_h=0.01,  # 原0.015 → 减少色调扰动
        hsv_s=0.5,  # 原0.6 → 降低饱和度扰动
        hsv_v=0.3,  # 原0.4 → 减少亮度变化

        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0, # 变形对数字不友好
        perspective=0.0005,  # 原0.001 → 降低透视变形强度
        flipud=0.0,  # 保持禁用 这个是上下翻转
        fliplr=0.0,  # 保持禁用 这个是左右翻转
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # 日志和输出
        project =  PROJECT_ROOT / "runs_digit",
        name = log_name,
        verbose = True,
    )


if __name__ == "__main__":
    train_yolo8()
    # tensorboard --logdir D:\Projects\dart-score-ai\runs_digit