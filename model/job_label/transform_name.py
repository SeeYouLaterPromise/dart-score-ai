import os



def transform_name(img_dir, start):
  for i, file in enumerate(os.listdir(img_dir)):
    if file.endswith(".txt"):
        old_path = os.path.join(img_dir, file)
        new_file = f"IMG_{start + i:04d}.txt"
        new_path = os.path.join(img_dir, new_file)
        os.rename(old_path, new_path)
        print(f"重命名: {file} -> {new_file}")
    


if __name__ == "__main__":
  img_dir = "job100/labels/train/d1_02_04_2020"
  start = 1231
  transform_name(img_dir, start)
