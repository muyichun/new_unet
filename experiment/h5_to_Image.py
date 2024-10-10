import h5py
from PIL import Image
import numpy as np





# 打开HDF5文件
with h5py.File('/Users/muyichun/Desktop/3_210/实验数据/camdata__valid.hdf5', 'r') as f:
    images = f['Untitled']
    for i, img in enumerate(images):
        pil_image = Image.fromarray(img)
        # 保存图像
        pil_image.save(f'/Users/muyichun/Desktop/3_210/实验数据/speckle_valid_128/{i}.png')
