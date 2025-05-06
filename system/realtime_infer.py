import cv2
import time
from pathlib import Path
import numpy as np
from evaluate_dart_image import evaluate_dart_image
from model.predict_darts import predict_image, visualize

# ========= 参数配置 =========
CAMERA_ID = 0
CROP_BOX = None  # [x1, y1, x2, y2] 手动选择后保存

# ========= 鼠标框选回调 =========
refPt = []


def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI", param)


# ========= 视频流捕获与预测 =========
def run_camera():
    global CROP_BOX
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("✅ 摄像头已启动，按 [r] 框选区域，按 [p] 预测，按 [q] 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取失败")
            break

        disp = frame.copy()

        # 显示裁剪框
        if CROP_BOX:
            x1, y1, x2, y2 = CROP_BOX
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Live", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            print("🔍 请拖动鼠标框选 ROI")
            tmp = frame.copy()
            cv2.imshow("Select ROI", tmp)
            cv2.setMouseCallback("Select ROI", click_and_crop, param=tmp)
            cv2.waitKey(0)
            cv2.destroyWindow("Select ROI")
            if len(refPt) == 2:
                x1, y1 = refPt[0]
                x2, y2 = refPt[1]
                CROP_BOX = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                print(f"✅ ROI 选定为：{CROP_BOX}")
        elif key == ord("p") and CROP_BOX:
            x1, y1, x2, y2 = CROP_BOX
            crop = frame[y1:y2, x1:x2]

            xy, processed_img = predict_image(crop.copy())
            print(xy)
            vis = visualize(processed_img.copy(), xy)
            cv2.imshow("annotate", vis)
            # score, total_score, labeled_img = evaluate_dart_image(crop.copy())
            # cv2.imshow("Predict", labeled_img)
            # print(f"📍 检测坐标点：{xy}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
