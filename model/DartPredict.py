import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import global_config
from ultralytics import YOLO
import torch
import cv2


class DartPredict:
  def __init__(self, ):
    self.model_path = global_config.get_model_weight_path()
    self.img_size = global_config.get_input_size()
    self.conf_thres = global_config.get_threshold()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.yolo_ultra = YOLO(self.model_path).to(self.device)


  def predict_image(self, image):
    results = self.yolo_ultra.predict(
        source=image,
        imgsz=self.img_size,
        conf=self.conf_thres,
        iou=0.45,  # 如果两个预测框的 IOU 大于 0.45，则认为它们是重复的，删除置信度较低的框。
        device=self.device,
        verbose=False
    )  
    
    # 确保结果数组长度为1, 一张图片对应一个result => 可以写成多线程，异步调用多帧处理？
    # 我突然想到可以这样：现在飞镖没有扎上去之前先预测一次得到校准点，然后再预测飞镖位置。
    assert len(results) == 1, f"Results array length is not 1, actual length: {len(results)}"
    
    # 取第一张图的结果
    result = results[0]
    boxes = result.boxes
    cls = boxes.cls.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()

    board_pts = []
    dart_pts = []
    xy = []
    h, w = image.shape[:2]

    for i, box in enumerate(coords):
        cls_id = int(cls[i])
        x_center = (box[0] + box[2]) / 2 / w
        y_center = (box[1] + box[3]) / 2 / h
        if cls_id < 4:  # Board1-4
            board_pts.append([cls_id, x_center, y_center])
        elif cls_id == 4:  # Dart
            dart_pts.append([x_center, y_center])

    board_pts.sort(key=lambda x: x[0])
    board_pts = [[x, y] for _, x, y in board_pts]

    xy.extend(board_pts)
    xy.extend(dart_pts)
    return xy, image
  
  # input: single image and coordinate points
  # output: original image, drawed image
  def visualize(image, xy):
    h, w = image.shape[:2]
    for i, (x, y) in enumerate(xy):
        cx, cy = int(x * w), int(y * h)
        color = (0, 255, 0) if i < 4 else (0, 0, 255)
        vis = image.copy()
        circle_radius = global_config.get_draw_radius()
        cv2.circle(vis, (cx, cy), circle_radius, color, -1)
        cv2.putText(vis, f"{i+1}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image, vis



if __name__ == "__main__":
  dart_predict = DartPredict()