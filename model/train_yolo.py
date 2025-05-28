from yolov5 import train
import os
from pathlib import Path
from ultralytics import YOLO

os.environ["YOLO_OVERWRITE_CACHE"] = "1"    # 强制覆盖缓存
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # solve: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]

weight_path = PROJECT_ROOT / "runs_dart" / "yolov5_hyp" / "weights" / "best.pt"

data_path = PROJECT_ROOT / "config" / "data_dart_digit.yaml"
project_path = PROJECT_ROOT / "runs_digit"
hyp_path = PROJECT_ROOT / "config/dart_hyp.yaml"
cfg_path = PROJECT_ROOT / "config" / "yolov5_dart_digit.yaml"

def train_yolo5(log_name="yolov5_digit"):
    train.run(
        img=640,
        batch=4,
        epochs=100,
        data=str(data_path),
        weights=str(weight_path),
        hyp=hyp_path,
        cfg=cfg_path,
        project=str(project_path),
        name=log_name,
        freeze=list(range(10)),  # 更简洁
        # cache=True  # ✅ 可选加速
        augment=False,  # ✅ 禁用增强器
        workers=0,  # ✅ 禁用多进程避免 memory error
        cache=False,  # ✅ 减少 cache 占
    )

def train_yolo8(log_name):
    # load the model
    model = YOLO(weight_path)

    # Train the model
    results = model.train(
        batch=16,
        epochs=2,
        data=str(data_path),
        weights=str(weight_path),
        hyp=hyp_path,
        cfg=cfg_path,
        project=str(project_path),
        name=log_name
    )

    # === Accessing Results ===
    print("\nTraining Losses:")
    print(f"Box Loss: {results.box_loss:.4f}")
    print(f"Class Loss: {results.cls_loss:.4f}")

    print("\nValidation Metrics:")
    print(f"mAP@50: {results.metrics['map50']:.4f}")
    print(f"mAP@50-95: {results.metrics['map']:.4f}")

    print("\nModel Speed:")
    print(f"Inference Speed: {results.speed['inference']:.2f} ms/img")


if __name__ == "__main__":
    train_yolo5()