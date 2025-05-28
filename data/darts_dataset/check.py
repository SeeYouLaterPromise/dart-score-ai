import os

label_dir = './data/darts_dataset/50/labels/train'
for file in os.listdir(label_dir):
    if file.endswith('.txt'):
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id >= 8:
                        print(f"❌ 文件 {file} 第{i+1}行：非法类别ID {cls_id}")
print("✅ 检查完成，没有发现非法类别ID。")
