import cv2
from model.predict_darts import predict_image, visualize
from model.dart_scoring import calculate_dart_scores, label_dart_scores

def evaluate_dart_image(img_path, model_path=None, show_steps=False):
    """
    输入图片路径，返回得分结果和标记后的图片
    参数:
        img_path: 输入图片路径
        model_path: 模型权重文件路径（若为None则使用默认路径）
        show_steps: 是否显示中间处理步骤的图片
    返回:
        scores: 得分结果列表
        total_score: 总得分
        labeled_image: 标记后的图片
    """
    # 1. 读取图片并进行预测
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"图片未找到：{img_path}")
    
    # 2. 预测关键点和飞镖点
    xy, processed_img = predict_image(image)
    
    # 3. 分离校准点和飞镖点
    calibration_points = xy[:4]
    dart_points = xy[4:]
    
    # 4. 计算得分
    scores, total_score = calculate_dart_scores(calibration_points, dart_points)
    
    # 5. 在图片上标记得分
    vis_image = visualize(processed_img.copy(), xy)
    labeled_image = label_dart_scores(vis_image, dart_points, scores)
    
    # 可选：显示中间步骤
    if show_steps:
        cv2.imshow("原始图片", processed_img)
        cv2.imshow("标记点", vis_image)
        cv2.imshow("得分结果", labeled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return scores, total_score, labeled_image

# 示例使用
if __name__ == "__main__":
    img_path = "../data/darts_dataset/800/d1_02_04_2020/IMG_1082.JPG"  # 替换为实际路径
    scores, total, labeled_img = evaluate_dart_image(img_path, show_steps=True)
    
    print("\n得分详情：")
    for i, score in enumerate(scores):
        print(f"飞镖 {i+1}: {score['details']}")
    print(f"总得分: {total}")
    
    cv2.imwrite("result_labeled.jpg", labeled_img)
