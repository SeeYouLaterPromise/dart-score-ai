import cv2
import time
from pathlib import Path
import numpy as np
from evaluate_dart_image import evaluate_dart_image
from model.predict_darts import predict_image, visualize
import cv2

img_source = mg_source = cv2.imread("result_labeled.jpg")  # 数据集中有“BLADE”字体的图

def paste_patch_on_dartboard(img_target):
    # 读取图像
    # img_target = cv2.imread(dart_img_path)  # 你们拍摄的图（目标图）
    img_target = cv2.resize(img_target, (800, 800))
    # img_source = cv2.imread(font_patch_path)  # 数据集中有“BLADE”字体的图

    # --- 1. 裁剪源图中下方区域的字体片段 ---
    h_src, w_src = img_source.shape[:2]
    font_patch = img_source[h_src - 100:h_src - 20, int(w_src * 0.25):int(w_src * 0.75)]  # 中间部分，避免边角

    # --- 2. Resize patch 到目标图的底部宽度 ---
    h_tgt, w_tgt = img_target.shape[:2]
    patch_resized = cv2.resize(font_patch, (w_tgt // 2, font_patch.shape[0]))

    # --- 3. 粘贴 patch 到目标图的下方中央 ---
    result = img_target.copy()
    ph, pw = patch_resized.shape[:2]
    x_offset = (w_tgt - pw) // 2
    y_offset = h_tgt - ph - 10  # 上移10像素，避免贴太底

    # 替换区域
    result[y_offset:y_offset + ph, x_offset:x_offset + pw] = patch_resized

    return result

def extract_dartboard_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
        param1=100, param2=30, minRadius=150, maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]

        # 先裁剪 ROI（注意边界检查）
        h, w = image.shape[:2]
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r), min(h, y + r)
        roi = image[y1:y2, x1:x2]

        # 构建 mask，圆心在裁剪后图像中心
        mask = np.zeros_like(roi)
        roi_center = (r if x - r >= 0 else x, r if y - r >= 0 else y)
        cv2.circle(mask, roi_center, r, (255, 255, 255), thickness=-1)

        # 应用 mask
        masked = cv2.bitwise_and(roi, mask)
        return roi

    return image



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
        disp = extract_dartboard_roi(disp)
        # disp = paste_patch_on_dartboard(disp)


        # 显示裁剪框
        if CROP_BOX:
            x1, y1, x2, y2 = CROP_BOX
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Live", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.imwrite("our.png", disp)
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
        elif key == ord("p"):
            crop = disp.copy()
            xy, processed_img = predict_image(crop.copy())
            print(xy)
            vis = visualize(processed_img.copy(), xy)
            cv2.imshow("annotate", vis)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
