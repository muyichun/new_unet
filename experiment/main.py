# Load imaging data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

base_dir = r'D:\XJ'
img_h5 = r'/Users/muyichun/Desktop/3_210/实验数据/camdata__test.hdf5'
Img_dim = 128


with h5py.File(img_h5, 'r') as f:
    print(list(f.keys()))
    dataset_name = list(f.keys())
    my_img_raw = [None] * len(dataset_name)
    print(len(dataset_name))
    for num_dset in range(len(dataset_name)):
        my_img_raw[num_dset] = f.get(dataset_name[num_dset])[()]

    my_check_img_raw = my_img_raw[0]
    img_num = my_check_img_raw.shape[0]
    img_frames = np.zeros((img_num, Img_dim, Img_dim), dtype=np.float64)
    my_check_img = np.zeros((my_check_img_raw.shape[0], my_check_img_raw.shape[1]), dtype=np.float64)

    for k in range(img_num):
        Img = np.zeros(np.prod(Img_dim ** 2), dtype=np.float64)
        my_check_img[k, :] = -my_check_img_raw[k, :] - np.min(-my_check_img_raw[k, :])
        Img[np.array([lut])] = my_check_img[k, :]
        img_frames[k, :, :] = Img.reshape(Img_dim, Img_dim)

    img_avg = np.mean(img_frames[0:my_check_img_raw.shape[0], :, :], axis=0)
    rt_img = np.rot90(img_avg)
    plt.rcParams['figure.figsize'] = [10, 10]
    c = plt.imshow(rt_img, cmap='gray')
    plt.colorbar()