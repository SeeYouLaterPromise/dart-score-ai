import numpy as np
import cv2
import math
import model
from model.predict_darts import predict_image, visualize

def calculate_dart_scores(calibration_points, dart_points):
    """
    计算飞镖得分
    输入:
        calibration_points: 4个校准点坐标列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dart_points: 飞镖坐标列表 [[x,y], ...]
    输出:
        results: 每个飞镖的得分列表 [{'score': 分数, 'details': 描述}, ...]
        total_score: 总得分
    """
    # 标准飞镖盘上的校准点（归一化半径1.0，双倍环外沿）
    # 按5&20->17&3->8&11->13&6顺序，角度为9°, 99°, 189°, 279°
    standard_points = np.float32([
        [math.cos(math.radians(351)), math.sin(math.radians(351))],   # 5&20
        [math.cos(math.radians(171)), math.sin(math.radians(171))],  # 17&3
        [math.cos(math.radians(261)), math.sin(math.radians(261))], # 8&11
        [math.cos(math.radians(81)), math.sin(math.radians(81))]  # 13&6
    ])

    # 输入校准点
    input_points = np.float32(calibration_points)

    # 计算单应性变换矩阵
    transform_matrix = cv2.getPerspectiveTransform(input_points, standard_points)

    # 飞镖扇区得分（从20开始顺时针）
    sectors = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    sector_angle = 360 / 20  # 每个扇区18°

    # 飞镖盘区域半径（归一化，基于双倍环外沿半径1.0）
    bullseye_inner = 0.03  # 内靶心
    bullseye_outer = 0.07  # 外靶心
    triple_ring_inner = 0.58
    triple_ring_outer = 0.62  # 三倍环
    double_ring_inner = 0.98
    double_ring_outer = 1.0   # 双倍环

    results = []
    total_score = 0

    for dart in dart_points:
        # 变换飞镖坐标
        dart_point = np.array([[dart]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(dart_point, transform_matrix)[0][0]
        x, y = transformed

        # 计算距离和角度
        distance = math.sqrt(x**2 + y**2)
        angle = math.degrees(math.atan2(y, x)) % 360  # 反正切计算角度

        # 确定扇区
        sector_idx = int((angle + sector_angle / 2) % 360 // sector_angle)
        base_score = sectors[sector_idx]

        # 确定区域和倍数
        if distance <= bullseye_inner:
            score = 50
            details = f"内靶心 (50分)"
        elif distance <= bullseye_outer:
            score = 25
            details = f"外靶心 (25分)"
        elif triple_ring_inner <= distance <= triple_ring_outer:
            score = base_score * 3
            details = f"三倍 {base_score} 区 ({score}分)"
        elif double_ring_inner <= distance <= double_ring_outer:
            score = base_score * 2
            details = f"双倍 {base_score} 区 ({score}分)"
        elif distance > double_ring_outer:
            score = 0
            details = f"脱靶 (0分)"
        else:
            score = base_score
            details = f"单倍 {base_score} 区 ({score}分)"

        results.append({'score': score, 'details': details})
        total_score += score

    return results, total_score

def label_dart_scores(image, dart_points, scores):
    """
    在飞镖盘图片上标记飞镖得分
    输入:
        image: numpy数组，飞镖盘图片 (800x800x3)
        dart_points: 飞镖坐标列表 [[x, y], ...]，归一化坐标
        scores: 得分列表 [{'score': 分数, 'details': 描述}, ...]
    输出:
        labeled_image: 标记了得分的numpy图片
    """
    # 复制图片以避免修改原图
    labeled_image = image.copy()

    # 图片尺寸
    height, width = image.shape[:2]

    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # 白色文本
    thickness = 2

    for i, (dart, score) in enumerate(zip(dart_points, scores)):
        # 将归一化坐标转换为像素坐标
        x, y = dart
        x_pixel = int(x * width)
        y_pixel = int(y * height)

        # 解析得分描述，提取区域和基数
        details = score['details']
        if "靶心" in details:
            label = "B" if "内靶心" in details else "OB"
        elif "脱靶" in details:
            label = "Out"
        else:
            # 提取倍数和基数
            multiplier = "S"
            if "双倍" in details:
                multiplier = "D"
            elif "三倍" in details:
                multiplier = "T"
            # 提取基数（假设格式为“倍数 基数 区”）
            base_score = int(details.split()[1])
            label = f"{multiplier}{base_score}"

        # 调整文本位置，稍微偏移飞镖点
        offset_x, offset_y = 20, 10
        text_position = (x_pixel + offset_x, y_pixel + offset_y)

        # 绘制文本背景框（可选，增强可读性）
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        box_coords = (
            (text_position[0], text_position[1] - text_height - baseline),
            (text_position[0] + text_width, text_position[1] + baseline)
        )
        cv2.rectangle(labeled_image, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

        # 在图片上绘制得分标签
        cv2.putText(labeled_image, label, text_position, font, font_scale, color, thickness)

    return labeled_image

def example(img_path):
    # print(f"📷 处理图像：{img_path.name}")
    # image = cv2.imread(str(img_path))
    image = cv2.imread(img_path)
    xy, img = predict_image(image)
    cv2.imshow("original", img)
    cv2.waitKey(0)

    print("预测点坐标：")
    for i, pt in enumerate(xy):
        print(f"  Point {i + 1}: {pt}")
    vis = visualize(img.copy(), xy)
    cv2.imshow("annotate", vis)
    cv2.waitKey(0)

    calibration = xy[:4]
    darts = xy[4:]

    scores, total = calculate_dart_scores(calibration, darts)
    for i, result in enumerate(scores):
        print(f"飞镖 {i + 1}: {result['details']}")
    print(f"总得分: {total}")

    labeled_image = label_dart_scores(vis, darts, scores)
    cv2.imshow("labeled", labeled_image)
    cv2.waitKey(0)



# 示例使用
if __name__ == "__main__":
    img_path = "patched_output2.jpg"
    example(img_path)