import cv2
import numpy as np

def paste_patch_on_dartboard(dart_img_path, font_patch_path, save_path="patched_output2.jpg"):
    # 读取图像
    img_target = cv2.imread(dart_img_path)  # 你们拍摄的图（目标图）
    img_target = cv2.resize(img_target, (800, 800))
    img_source = cv2.imread(font_patch_path)  # 数据集中有“BLADE”字体的图

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

    # --- 4. 显示并保存结果 ---
    cv2.imshow("with_font_patch", result)
    cv2.imwrite(save_path, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用示例
paste_patch_on_dartboard("our.png", "result_labeled.jpg")
