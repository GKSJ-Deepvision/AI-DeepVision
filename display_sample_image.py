import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

path_img_ex = "archive (2)\\ShanghaiTech\\part_B\\train_data\\images\\IMG_6.jpg"
image_ex = cv2.cvtColor(cv2.imread(path_img_ex),cv2.COLOR_BGR2RGB)
figure = plt.figure(figsize=(5,5))
plt.imshow(image_ex)
plt.show()

path_gt_ex = "archive (2)\\ShanghaiTech\\part_B\\train_data\\ground-truth\\GT_IMG_6.mat"
gt_ex = loadmat(path_gt_ex)
print('type: ', type(gt_ex))
print(gt_ex.items())
print(gt_ex.keys())

#plotting
figure = plt.figure(figsize=(5,5))
gt_coor_ex = gt_ex.get('image_info')[0][0][0][0][0]
print('Shape of coordinates: ', gt_coor_ex.shape)

print("Type:", type(gt_coor_ex))
print("Shape:", gt_coor_ex.shape)
print("First 5 coordinates:\n", gt_coor_ex[:5])

image_marked = image_ex.copy()
for x_cor, y_cor in gt_coor_ex:
    cv2.drawMarker(
        image_marked,
        (int(x_cor), int(y_cor)),
        color=(255, 0, 0),       # bright blue
        markerType=cv2.MARKER_CROSS,
        markerSize=25,           # much bigger
        thickness=4
    )

plt.figure(figsize=(6,6))
plt.imshow(image_marked)
plt.title("Image with Crowd Annotations")
plt.axis("off")
plt.show()
