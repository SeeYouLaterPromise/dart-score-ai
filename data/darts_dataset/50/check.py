import os


img_dir = "../../../data/darts_dataset/50"
label_dir = os.path.join(img_dir, "labels")

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".JPG")])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

img_basenames = {os.path.splitext(f)[0] for f in img_files}
label_basenames = {os.path.splitext(f)[0] for f in label_files}

missing_labels = img_basenames - label_basenames
extra_labels = label_basenames - img_basenames

print("缺失标签文件：", missing_labels)
print("多余标签文件：", extra_labels)
