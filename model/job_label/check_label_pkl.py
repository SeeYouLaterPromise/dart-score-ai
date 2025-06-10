import pandas as pd
import os

# === 设置 pkl 文件路径 ===
pkl_path = "../data/darts_dataset/labels.pkl"  # 请按你的实际路径修改

# === 加载 DataFrame ===
df = pd.read_pickle(pkl_path)

print("列名如下：")
print(df.columns)

# === 记录非法项 ===
error_rows = []

for i, row in df.iterrows():
    img_name = row['img_name']
    xy_list = row['xy']

    for j, (x, y) in enumerate(xy_list):
        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            print(f"❌ 非法坐标：索引={i} 文件={img_name} 点序={j} 坐标=({x:.6f}, {y:.6f})")
            error_rows.append(i)
            break  # 一张图只报一次错误

print(f"\n✅ 检查完成，共发现 {len(error_rows)} 条含非法坐标的标注记录。")

