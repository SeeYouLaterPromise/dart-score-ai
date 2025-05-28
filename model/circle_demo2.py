import cv2
import numpy as np

def detect_outer_circle_refined(img,
                                top_k=5,
                                min_area_ratio=0.1,
                                max_area_ratio=0.9,
                                circ_thresh=0.7,
                                ar_thresh=0.2):
    """
    1) sea‐level: Canny + morphology to get edges
    2) shortlist: take top_k largest contours
    3) filter: keep only those with area_ratio, circularity, aspect ratio OK
    4) merge & convexHull, minEnclosingCircle
    """

    h, w = img.shape[:2]
    img_area = h * w

    # 1. 边缘检测 + 闭运算 + 膨胀
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=2)

    # 2. 找所有外轮廓
    contours, _ = cv2.findContours(dilated,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("没有检测到任何轮廓")

    # 3. 海选：面积最大 top_k
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]

    good_pts = []
    # 4. 复选：面积比、圆度、长宽比
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        circ = 4 * np.pi * area / (peri * peri)
        if circ < circ_thresh:
            continue

        x,y,wc,hc = cv2.boundingRect(cnt)
        ar = float(wc) / hc
        if abs(ar - 1.0) > ar_thresh:
            continue

        # 通过筛选，保留这个轮廓的所有点
        good_pts.append(cnt.reshape(-1,2))
    
    if not good_pts:
        # 回退到 top_k 融合
        good_pts = [c.reshape(-1,2) for c in contours]

    # 5. 融合 & 凸包 & 拟合圆
    all_pts = np.vstack(good_pts)
    hull = cv2.convexHull(all_pts)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    return int(cx), int(cy), int(r)


# —— 用法示例 —— 
img_path = "system/data_20250513_152524.png"
angle_shit = 10
rough_ratio = 0.75


img = cv2.imread(img_path)
Cx, Cy, R = detect_outer_circle_refined(img)

# 可视化检查
vis = img.copy()
cv2.circle(vis, (Cx, Cy), R, (0, 0, 255), 3)     # 红色最终拟合圆
cv2.circle(vis, (Cx, Cy), 5, (0, 255, 255), -1)  # 黄色圆心

# 四校准点
for ang in [90, 0, 270, 180]:
    theta = np.deg2rad(ang)
    px = int(Cx + 0.93*R * np.cos(theta))
    py = int(Cy - 0.93*R * np.sin(theta))
    cv2.circle(vis, (px, py), 8, (0, 255, 0), 2)

cv2.imshow("Result", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
