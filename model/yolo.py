from ultralytics import YOLO
from pathlib import Path

# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
print(PROJECT_ROOT)
# Load a model
model = YOLO("yolov8s.pt")  # build a new model from YAML

data_path = PROJECT_ROOT / "config/yolo_data.yaml"
cfg_path = PROJECT_ROOT / "config/dart_hyp.yaml"
# Train the model
results = model.train(data=data_path, epochs=10, imgsz=640, batch=16, cfg=cfg_path, name="dart-aug-v1", project="runs_dart")
