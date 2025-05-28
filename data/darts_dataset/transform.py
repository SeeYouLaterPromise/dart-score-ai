import os

img_dir = "data/darts_dataset/50"
for file in os.listdir(img_dir):
    if file.endswith(".JPG"):
        old_path = os.path.join(img_dir, file)
        new_file = file.replace(".JPG", ".jpg")
        new_path = os.path.join(img_dir, new_file)
        os.rename(old_path, new_path)
        print(f"重命名: {file} -> {new_file}")
