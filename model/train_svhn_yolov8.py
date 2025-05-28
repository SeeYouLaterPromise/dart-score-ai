from ultralytics import YOLO
import os
from pathlib import Path


os.environ["YOLO_OVERWRITE_CACHE"] = "1"    # 强制覆盖缓存
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # solve: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]

weight_path = PROJECT_ROOT / "model" / "yolov8s.pt"

data_path = PROJECT_ROOT / "config" / "data_svhn.yaml"
project_path = PROJECT_ROOT / "runs_digit"
hyp_path = PROJECT_ROOT / "config/dart_hyp.yaml"
cfg_path = PROJECT_ROOT / "config" / "yolov5s_svhn.yaml"

# 初始化模型（可换成 yolov8m.pt/yolov8l.pt/yolov8x.pt）
model = YOLO("yolov8s.pt")

# 启动训练
model.train(
    data=data_path,  # 路径需指向你的 SVHN 数据配置文件
    # model="yolov8s.yaml",  # 你的自定义 model 架构文件（包含 anchors、backbone、head）
    epochs=100,
    imgsz=640,
    batch=32,
    device=1,  # 设置为 'cpu' 使用 CPU，或具体 GPU 编号如 0
    workers=4,
    optimizer="SGD",  # 也可用 'Adam', 'AdamW'
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=50,

    # 数据增强参数
    degrees=5.0,
    translate=0.05,
    scale=0.3,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,

    # 日志和输出
    name="yolov8_svhn",
    project=project_path,
    verbose=True
)
