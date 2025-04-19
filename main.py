import pandas as pd
import cv2
import os

image_dir = "./data/darts_dataset/800"

sum = 0
for d_dir in os.listdir(image_dir):
    new_dir = os.path.join(image_dir, d_dir)
    num_image = len(os.listdir(new_dir))
    sum += num_image
print(sum)


label = pd.read_pickle("./data/darts_dataset/labels.pkl")
print(len(label))
row = label.iloc[1]
print(row['img_folder'])
img_path = f"./data/darts_dataset/800/{row['img_folder']}/{row['img_name']}"
img = cv2.imread(img_path)

for i, (x, y) in enumerate(row['xy']):
    px, py = int(x * 800), int(y * 800)
    # for reference point, color it green.
    # for dart point, color it blue.
    color = (0, 255, 0) if i < 4 else (0, 0, 255)
    cv2.circle(img, (px, py), 6, color, -1)

cv2.imshow("label visual", img)
cv2.waitKey(0)
