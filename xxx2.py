import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
# image_path = '/Users/muyichun/Downloads/SharedData/Figure2/X_shared_Figure_2.npy'
# image_path = '/Users/muyichun/Downloads/SharedData/Figure3and4/MNIST_Figure_3_first_order.npy'
# image_path = '/Users/muyichun/Downloads/SharedData/Figure3and4/mnist_shared_label_Figure_3_and_4.npy'
image_data = np.load(image_path)

# 检查图像数据的形状
print(f"Image shape: {image_data.shape}")


# 显示图像
plt.imshow(image_data[222], cmap='gray')  # 如果是灰度图，使用 'gray' colormap
plt.axis('off')  # 可选：关闭坐标轴
plt.show()