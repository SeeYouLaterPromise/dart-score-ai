import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import platform

# 兼容 Windows 路径
os_type = platform.system()
if os_type == 'Linux':
    import pathlib
    pathlib.WindowsPath = pathlib.PosixPath

# ========== 路径设置 ==========
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))
YOLOV5_ROOT = PROJECT_ROOT / 'model' / 'yolov5'
sys.path.insert(0, str(YOLOV5_ROOT))

# ========== YOLOv5 模块导入 ==========
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.general import non_max_suppression, scale_boxes
from model.yolov5.utils.augmentations import letterbox
from model.yolov5.utils.torch_utils import select_device

# ========== 参数设置 ==========
MODEL_PATH = PROJECT_ROOT / 'runs_dart' / 'yolov5_third' / 'weights' / 'best.pt'
# IMAGE_DIR = PROJECT_ROOT / 'data' / 'darts_dataset' / '50' / 'images' / 'train'
IMAGE_DIR = PROJECT_ROOT / "system" / "images"
fmt = 'png '
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
DEVICE = select_device('0' if torch.cuda.is_available() else 'cpu')

# ========== 模型加载 ==========
model = DetectMultiBackend(str(MODEL_PATH), device=DEVICE)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))

# ========== 推理函数 ==========
def predict_image(image):
    img0 = image.copy()
    img = letterbox(img0, new_shape=IMG_SIZE, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)

    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f'{names[int(cls)]} {conf:.2f}'
                results.append(((x1, y1, x2, y2), label, int(cls)))
    return results, img0

# ========== 可视化函数 ==========
def visualize(image, results):
    for (x1, y1, x2, y2), label, cls_id in results:
        color = (0, 255, 0) if cls_id < 4 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x2 - 5, y2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# ========== 示例测试函数 ==========
def example():
    for img_path in sorted(IMAGE_DIR.glob(f"*.{fmt}")):
        print(f"📷 Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        results, img = predict_image(image)

        print("🔍 Results:")
        for box, label, _ in results:
            print(f"  - {label} at {box}")

        vis = visualize(img.copy(), results)
        cv2.imshow("Prediction", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

# ========== 主程序 ==========
if __name__ == "__main__":
    example()
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import platform

# 兼容 Windows 路径
os_type = platform.system()
if os_type == 'Linux':
    import pathlib
    pathlib.WindowsPath = pathlib.PosixPath

# ========== 路径设置 ==========
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))
YOLOV5_ROOT = PROJECT_ROOT / 'model' / 'yolov5'
sys.path.insert(0, str(YOLOV5_ROOT))

# ========== YOLOv5 模块导入 ==========
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.general import non_max_suppression, scale_boxes
from model.yolov5.utils.augmentations import letterbox
from model.yolov5.utils.torch_utils import select_device

# ========== 参数设置 ==========
MODEL_PATH = PROJECT_ROOT / 'runs_digit' / 'yolov5_digit6' / 'weights' / 'best.pt'
# IMAGE_DIR = PROJECT_ROOT / 'data' / 'yolo_dataset' / 'images' / 'val'
IMAGE_DIR = PROJECT_ROOT / 'system'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
DEVICE = select_device('0' if torch.cuda.is_available() else 'cpu')

# ========== 模型加载 ==========
model = DetectMultiBackend(str(MODEL_PATH), device=DEVICE)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))

# ========== 推理函数 ==========
def predict_image(image):
    img0 = image.copy()
    img = letterbox(img0, new_shape=IMG_SIZE, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)

    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f'{names[int(cls)]} {conf:.2f}'
                results.append(((x1, y1, x2, y2), label, int(cls)))
    return results, img0

# ========== 可视化函数 ==========
def visualize(image, results):
    for (x1, y1, x2, y2), label, cls_id in results:
        color = (0, 255, 0) if cls_id < 4 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x2 - 5, y2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# ========== 示例测试函数 ==========
def example():
    for img_path in sorted(IMAGE_DIR.glob("*.png")):
        print(f"📷 Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        results, img = predict_image(image)

        print("🔍 Results:")
        for box, label, _ in results:
            print(f"  - {label} at {box}")

        vis = visualize(img.copy(), results)
        cv2.imshow("Prediction", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

# ========== 主程序 ==========
if __name__ == "__main__":
    example()
