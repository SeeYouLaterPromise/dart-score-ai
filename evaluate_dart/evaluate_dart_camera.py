import cv2
import time
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

def evaluate_dart_camera(model_path=None, show_steps=False):
    """
    使用摄像头实时评估飞镖得分
    参数:
        model_path: 模型权重文件路径（若为None则使用默认路径）
        show_steps: 是否显示中间处理步骤的图片
    """
    cap = cv2.VideoCapture(0)  # 使用默认摄像头（索引0）
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        start_time = time.time()

        try:
            # 运行完整的处理流程
            scores, total, labeled_img = evaluate_dart_image(None, model_path=model_path, show_steps=False)
        except Exception as e:
            print(f"处理帧时出错: {e}")
            labeled_img = frame.copy()
            cv2.putText(labeled_img, "处理中...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示处理后的帧
        cv2.imshow('实时飞镖得分识别', labeled_img)

        duration = time.time() - start_time
        fps = 1 / duration if duration > 0 else 0
        print(f"处理帧耗时: {duration:.3f}秒, FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 示例使用
if __name__ == "__main__":
    # 启动摄像头模式
    evaluate_dart_camera(show_steps=True)
