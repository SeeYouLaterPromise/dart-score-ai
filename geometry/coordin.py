import cv2
import matplotlib.pyplot as plt

img_path = "data/darts_dataset/d1_02_04_2020/IMG_1081.JPG"
vis = cv2.imread(img_path)
vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6,6))
plt.imshow(vis_rgb)
plt.axis('off')
plt.show()
