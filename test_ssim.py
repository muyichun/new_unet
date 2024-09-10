import os

import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from UNet1.architect.utils import keep_image_size_open

imgsPath1 = '/Users/muyichun/Desktop/3_210/fz_testLabel'
# imgsPath1 = '/Users/muyichun/Desktop/3_210/shouxie_label_test_128'
imgsPath2 = '/Users/muyichun/Desktop/3_210/my_test_predict'
imgs_dir = os.listdir(imgsPath1)
max = 0
x = list(range(20701, 20797))
# x = list(range(29301, 29379))
y = []
for i in x:
    image1 = keep_image_size_open(imgsPath1 + '/' + str(i) + ".png")
    image2 = keep_image_size_open(imgsPath2 + '/' + str(i) + ".png")
    ssim_score = ssim(np.array(image1), np.array(image2))
    if ssim_score > max:
        max = ssim_score
    print(f"The SSIM score is {ssim_score}", " xxx ", i)
    y.append(ssim_score)
print(f"The max SSIM score is {max}")

# 绘制散点图
plt.scatter(x, y)
plt.title('SSIM distribution')
plt.xlabel('Number of pictures')
plt.ylabel('SSIM')
plt.show()