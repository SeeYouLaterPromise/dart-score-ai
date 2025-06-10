import cv2
from ultralytics import YOLO
import os
import numpy as np
import sys
from pathlib import Path

# ========== 路径设置 ==========
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))
YOLOV5_ROOT = PROJECT_ROOT / 'model' / 'yolov5'
sys.path.insert(0, str(YOLOV5_ROOT))

# ========== YOLOv5 模块导入 ==========
from model.yolov5.utils.general import scale_boxes

weights_path = 'runs_dart/yolov11m_first/weights/best.pt'
# image_dir = '../data/yolo_dataset/images/val'
image_dir = "system/data"
# output_dir = 'model/svhn_results'
conf_threshold = 0.25
save_result = False
# os.makedirs(output_dir, exist_ok=True)
model = YOLO(weights_path)

def predict_yolov8(image):
    img = image.copy()

    # 推理
    results = model.predict(source=img, conf=conf_threshold, save=False, verbose=False)
    result = results[0] if len(results) != 0 else None
    assert result is not None

    img_pred = img.copy()
    boxes = result.boxes
    cls_names = result.names

    for box in boxes:
        # 获取边框与标签
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        label = f'{cls_names[cls_id]} {conf:.2f}'

        # 绘制边框与标签
        cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_pred, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 放大显示（可调节）
    img_pred = cv2.resize(img_pred, (800, 800))
    return img, results, img_pred


def get_points(pred, img0):
    xy = []
    for det in pred:  # 遍历每一张图
        if len(det):
            det[:, :4] = scale_boxes(img0.shape[2:], det[:, :4], img0.shape).round()
            h, w = img0.shape[:2]
            board_pts = det[det[:, 5] < 4].cpu().numpy()
            dart_pts = det[det[:, 5] == 4].cpu().numpy()

            board_pts = board_pts[np.argsort(board_pts[:, 5])]  # 按 class 排序
            for box in np.vstack((board_pts, dart_pts)):
                x_center = (box[0] + box[2]) / 2 / w
                y_center = (box[1] + box[3]) / 2 / h
                xy.append([x_center, y_center])
    return xy, img0


def predict_image(image):
    img0, results, _ = predict_yolov8(image)
    return get_points(results, img0)


if __name__ == "__main__":
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        orig_h, orig_w = img.shape[:2]

        # 推理
        results = model.predict(source=img, conf=conf_threshold, save=False, verbose=False)

        for result in results:
            print(f"here, {len(results)}")
            img_pred = img.copy()
            boxes = result.boxes
            cls_names = result.names

            for box in boxes:
                # 获取边框与标签
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = f'{cls_names[cls_id]} {conf:.2f}'

                # 绘制边框与标签
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_pred, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 放大显示（可调节）
            img_pred = cv2.resize(img_pred, (800, 800))

            while True:
                # print("here, while true")
                cv2.imshow("Prediction", img_pred)
                key = cv2.waitKey(0) & 0xFF

                if key == ord("q"):
                    print("🛑 程序已退出")
                    cv2.destroyAllWindows()
                    exit(0)

                elif key == ord("s"):
                    # save_path = os.path.join(output_dir, image_name)
                    # cv2.imwrite(save_path, img_pred)
                    # print(f"💾 已保存预测结果图：{image_name}")
                    break

                elif key == ord("l"):
                    break

                else:
                    print("⚠️ 按下无效按键，请按 'q' 退出，'s' 保存，或 'l' 查看下一张")
