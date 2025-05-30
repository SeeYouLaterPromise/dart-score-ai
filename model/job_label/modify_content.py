import re

# 输入输出文件路径
input_file = 'job100/output.txt'     # 原始文件
output_file = 'job100/output.txt'   # 修改后的文件

# 读取并修改内容
with open(input_file, 'r') as f:
    lines = f.readlines()

modified_lines = []
for line in lines:
    # 提取IMG_后的数字
    match = re.search(r'IMG_(\d+)\.jpg', line)
    if match:
        original_num = int(match.group(1))
        new_num = original_num + 150
        # 构造新路径
        new_line = re.sub(r'IMG_\d+\.jpg', f'IMG_{new_num}.jpg', line)
        modified_lines.append(new_line)
    else:
        modified_lines.append(line)  # 如果不匹配就原样保留

# 写入新文件
with open(output_file, 'w') as f:
    f.writelines(modified_lines)

print("✅ 修改完成，已保存到", output_file)
