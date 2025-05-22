import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_outer_circle_strict(img):
    # 1. 预处理：灰度 + 高斯
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Canny 边缘
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)

    # 3. 形态学闭运算：填补边缘断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    cv2.imshow("closed", closed)
    cv2.waitKey(0)

    # 4. 找最大轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)

    # 5. 取凸包，再拟合外接圆
    hull = cv2.convexHull(max_cnt)
    (x, y), r = cv2.minEnclosingCircle(hull)
    return int(x), int(y), int(r)


if __name__ == "__main__":
    standard_img_name = "result_labeled.jpg"
    environment_img_name = "predict_20250513_163734.png"
    img_path = f"system/{environment_img_name}"
    angle_shit = 10
    rough_ratio = 0.75


    img = cv2.imread(img_path)
    Cx, Cy, R = detect_outer_circle_strict(img)

    # 可视化
    vis = img.copy()
    cv2.circle(vis, (Cx, Cy), R, (0, 0, 255), 2)     # 红色拟合圆
    cv2.circle(vis, (Cx, Cy), 5, (0, 255, 255), -1)  # 黄色圆心

    # 四个校准点
    angles = [90, 0, 270, 180]  # 上、右、下、左
    angles = [angle + angle_shit for angle in angles]
    r_calib = int(rough_ratio * R)
    for ang in angles:
        th = np.deg2rad(ang)
        px = int(Cx + r_calib * np.cos(th))
        py = int(Cy - r_calib * np.sin(th))
        cv2.circle(vis, (px, py), 28, (0, 255, 0), 2)

    # 显示
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.imshow(vis_rgb)
    plt.axis('off')
    plt.show()
    # cv2.imwrite(f"./model/circle_demo3.jpg", vis)
