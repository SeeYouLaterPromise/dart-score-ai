import sys
import torch
import cv2
import numpy as np
from pathlib import Path

import platform
os_type = platform.system()
print("Operating System:", os_type)
# If you have linux (or deploying for linux) use:
if os_type == 'Linux':
    from pathlib import Path
    import pathlib
    temp = pathlib.PosixPath
    pathlib.WindowsPath = pathlib.PosixPath


# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))
YOLOV5_ROOT = PROJECT_ROOT / 'model' / 'yolov5'
sys.path.insert(0, str(YOLOV5_ROOT))  # 把 yolov5 路径放在最前面！


# ✅ 使用 yolov5.xxx 模块路径
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.general import non_max_suppression, scale_boxes
from model.yolov5.utils.augmentations import letterbox

# === 参数配置 ===
print(FILE.parents[0])
print(FILE.parents[1])
print(FILE.parents[2])
MODEL_PATH = PROJECT_ROOT / 'runs_darts' / 'yolov5_third9' / 'weights' / 'best.pt'
IMAGE_DIR = PROJECT_ROOT / 'data' / 'yolo_dataset' / 'images' / 'val'
CONF_THRESHOLD = 0.4
IMG_SIZE = 800
# 这样传给 DetectMultiBackend(..., device=DEVICE) 的就是 torch.device('cuda') 或 torch.device('cpu') ✅
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 加载模型 ===
model = DetectMultiBackend(str(MODEL_PATH), device=DEVICE)
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))  # 预热模型

# === 图像推理函数 ===
def predict_image(image):
    img0 = image.copy()
    img = letterbox(img0, new_shape=IMG_SIZE, stride=stride)[0]
    # cv2.imshow("resized", img)
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(DEVICE).float()
    img /= 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)

    for i in range(len(pred)):
        p = pred[i]
        if type(p) is torch.Tensor:
            print(f"{i}-index: Tensor type: {p.shape}")
        elif type(p) is list:
            print(f"{i}-index: List length: {len(p)}")
            for c_p in p:
                print(f"type: {type(c_p)}")
    # print(f"len(pred): {len(pred)}")
    # print(f"pred[0].shape: {pred[0].shape}")
    # print(f"list pred[1]: {len(pred[1])}")

    for i, t in enumerate(pred[1]):
        print(f"pred[1][{i}] shape: {t.shape}")


    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=0.45)

    xy = []
    for det in pred:  # 遍历每一张图
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            h, w = img0.shape[:2]
            board_pts = det[det[:, 5] < 4].cpu().numpy()
            dart_pts = det[det[:, 5] == 4].cpu().numpy()

            board_pts = board_pts[np.argsort(board_pts[:, 5])]  # 按 class 排序
            for box in np.vstack((board_pts, dart_pts)):
                x_center = (box[0] + box[2]) / 2 / w
                y_center = (box[1] + box[3]) / 2 / h
                xy.append([x_center, y_center])
    return xy, img0

# === 可视化函数 ===
def visualize(image, xy):
    h, w = image.shape[:2]
    for i, (x, y) in enumerate(xy):
        cx, cy = int(x * w), int(y * h)
        color = (0, 255, 0) if i < 4 else (0, 0, 255)
        cv2.circle(image, (cx, cy), 6, color, -1)
        cv2.putText(image, f"{i+1}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def example():
    for img_path in sorted(IMAGE_DIR.glob("*.jpg")):
        print(f"📷 处理图像：{img_path.name}")
        image = cv2.imread(str(img_path))
        xy, img = predict_image(image)

        # cv2.imshow("original", img)
        # cv2.waitKey(0)

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

    cv2.imshow("original", img)
    cv2.waitKey(0)

    print("预测点坐标：")
    for i, pt in enumerate(xy):
        print(f"  Point {i+1}: {pt}")
    vis = visualize(img.copy(), xy)
    cv2.imshow("annotate", vis)
    cv2.waitKey(0)

def check_model():
    # print(model.model)               # 查看整个结构
    print(len(model.model.model))
    print(model.model.model[-1])     # 查看 Detect 层
    # print(model.model.yaml)          # 查看配置 yaml 内容
    # print(model.model.names)         # 类别名称

# === 主程序 ===
if __name__ == "__main__":
    # print("hello")
    example()
    # check_model()

