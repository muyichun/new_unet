import glob
import time

import cv2
import h5py
import numpy as np
import os
import natsort
import matplotlib.pyplot as plt


input_pic_dic = r'/Users/muyichun/Desktop/Demo/MCF_speckle_digits/label_128_副本2'
patterns_filename_to_save = r'/Users/muyichun/Desktop/Demo/MCF_speckle_digits/xxx/speckle_digits.hdf5'

file_names = [img for img in glob.glob(input_pic_dic+'/*.png')]
sorted_name = natsort.natsorted(file_names)
# Start the clock
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# Gernerate all the patterns
with h5py.File(patterns_filename_to_save, 'w') as ptn_f:
    dataset = ptn_f.create_dataset('Foci Patterns', shape=(0,624,128), maxshape=(None,624,128),dtype='uint8')
    for img in sorted_name:
        e = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_np = e[np.newaxis, :, :]  # 变为：(1, 128, 128)
        img_ff = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), np.complex64)
        img_ff = np.fft.fftshift(np.fft.fft2(img_np))

        nscan, nprox, nprox = img_np.shape
        Ny = 624
        Nx = 1024
        offset = (Nx-Ny)//2
        cx1 = 73
        cy1 = 334
        # cx1 = 326
        # cy2 = 79
        fields = np.zeros((nscan, Ny, Ny), np.complex64)
        fields[:, cx1-(nprox-0)//2:cx1+(nprox-0)//2, cy1-(nprox-0)//2:cy1+(nprox-0)//2] = img_ff

        pat = np.packbits(np.greater(np.angle(np.fft.ifft2(fields)), 0)).reshape(nscan, Ny, Ny//8)
        DMD_pat = np.zeros((nscan,Ny, Nx//8), dtype=np.uint8)
        DMD_pat[:,:,offset//8:(offset+Ny)//8] = pat
        # 追加写入新数据
        current_size = dataset.shape[0]
        print("写入中：", current_size)
        dataset.resize(current_size + DMD_pat.shape[0], axis=0)
        dataset[current_size:] = DMD_pat
# end the clock
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# # Todo: test
# img_iff = np.fft.ifft2(np.fft.ifftshift(img_ff))
# img_iff = np.fft.ifft2(img_ff)
# plt.imshow(np.abs(img_iff[0]))
# plt.show()