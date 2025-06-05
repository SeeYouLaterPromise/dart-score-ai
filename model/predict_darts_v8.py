import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import platform

print("here is predict_darts.py")

# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))

# === 参数配置 ===
MODEL_PATH = PROJECT_ROOT / 'runs_dart' / "yolov11n_first" / 'weights' / 'best.pt'
IMAGE_DIR = PROJECT_ROOT / 'data' / 'yolo_dataset' / 'images' / 'val'
CONF_THRESHOLD = 0.4
IMG_SIZE = 800

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === 加载模型（适配 YOLOv8）===
model = YOLO(str(MODEL_PATH)).to(DEVICE)

# === 图像推理函数（YOLOv8）===
def predict_image(image):
    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=0.45,
        device=DEVICE,
        verbose=False
    )[0]  # 取第一张图的结果

    xy = []
    h, w = image.shape[:2]

    boxes = results.boxes
    cls = boxes.cls.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()

    board_pts = []
    dart_pts = []

    for i, box in enumerate(coords):
        cls_id = int(cls[i])
        x_center = (box[0] + box[2]) / 2 / w
        y_center = (box[1] + box[3]) / 2 / h
        if cls_id < 4:  # Board1-4
            board_pts.append((cls_id, x_center, y_center))
        elif cls_id == 4:  # Dart
            dart_pts.append((x_center, y_center))

    board_pts.sort(key=lambda x: x[0])
    board_pts = [(x, y) for _, x, y in board_pts]

    xy.extend(board_pts)
    xy.extend(dart_pts)
    return xy, image

# === 可视化函数 ===
def visualize(image, xy):
    h, w = image.shape[:2]
    for i, (x, y) in enumerate(xy):
        cx, cy = int(x * w), int(y * h)
        color = (0, 255, 0) if i < 4 else (0, 0, 255)
        cv2.circle(image, (cx, cy), 2, color, -1)
        cv2.putText(image, f"{i+1}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# === 示例函数 ===
def example():
    for img_path in sorted(IMAGE_DIR.glob("*.jpg")):
        print(f"📷 处理图像：{img_path.name}")
        image = cv2.imread(str(img_path))
        xy, img = predict_image(image)

        print("预测点坐标：")
        for i, pt in enumerate(xy):
            print(f"  Point {i+1}: {pt}")
        vis = visualize(img.copy(), xy)
        cv2.imshow("annotate", vis)
        cv2.waitKey(0)
        break

def example_pic(img_path):
    print(f"📷 处理图像：{img_path.name}")
    image = cv2.imread(str(img_path))
    xy, img = predict_image(image)

    print("预测点坐标：")
    for i, pt in enumerate(xy):
        print(f"  Point {i+1}: {pt}")
    vis = visualize(img.copy(), xy)
    cv2.imshow("annotate", vis)
    cv2.waitKey(0)

# === 主程序 ===
if __name__ == "__main__":
    example()
