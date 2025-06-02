import os
import shutil

# 设置路径（你可以修改成自己的文件夹路径）
base_dir = "../../job100/labels/train/d1_02_04_2020/"
source_file = 'IMG_1182.txt'
start_index = 1183
end_index = 1230  # 包含1230
output_dir = base_dir  # 当前目录；你也可以改成其它目录路径

# 读取源文件内容
with open(os.path.join(output_dir, source_file), 'r') as f:
    content = f.read()

# 创建并写入48个目标文件
for idx in range(start_index, end_index + 1):
    target_filename = f'IMG_{idx}.txt'
    target_path = os.path.join(output_dir, target_filename)
    with open(target_path, 'w') as f:
        f.write(content)

print(f"成功复制 {end_index - start_index + 1} 份标注文件！")
