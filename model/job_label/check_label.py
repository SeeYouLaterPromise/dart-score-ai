import cv2
import os

# ==== 参数设置 ====
image_folder = "./data/darts_dataset/d1_02_04_2020"  # 图片和txt的所在目录
txt_folder = "./model/job_label/labels/train/d1_02_04_2020"  # 标注文件所在目录
start_idx = 1182
end_idx = 1230
image_ext = ".JPG"   # 可改为 .png 等格式
image_size = None    # 可手动指定为 (W, H)，否则自动读取

# ==== 播放每一帧 ====
for idx in range(start_idx, end_idx + 1):
    img_name = f"IMG_{idx}{image_ext}"
    txt_name = f"IMG_{idx}.txt"
    
    img_path = os.path.join(image_folder, img_name)
    txt_path = os.path.join(txt_folder, txt_name)

    # 读取图像
    if not os.path.exists(img_path):
        print(f"[跳过] 图像文件缺失：{img_path}")
        continue
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # 读取标注并绘制
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # 跳过格式错误行
                cls, cx, cy, bw, bh = map(float, parts)
                # 转换为绝对坐标
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                # 绘制矩形
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"cls {int(cls)}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print(f"[警告] 找不到标注文件：{txt_path}")

    # 显示图像
    cv2.imshow("YOLO Annotations Preview", image)
    key = cv2.waitKey(300)  # 播放速度（单位ms），按下任意键可提前跳转
    if key == 27:  # ESC键退出
        break

cv2.destroyAllWindows()
