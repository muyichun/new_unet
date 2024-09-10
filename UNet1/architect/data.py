import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from UNet1.architect.utils import my_image_open

'''
定义转换
'''
transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, root_path, img_name, label_name):
        self.root_path = root_path
        self.img_dir_name = img_name
        self.label_dir_name = label_name
        self.label_dir_list = os.listdir(os.path.join(self.root_path, self.label_dir_name))

    def __len__(self):
        return len(self.label_dir_list)

    def __getitem__(self, index):
        current_label_name = self.label_dir_list[index]    # 形式如：xx.png， 注意：当前图片名与对应标签名默认一致
        current_label_path = os.path.join(self.root_path, self.label_dir_name, current_label_name)
        current_image_path = os.path.join(self.root_path, self.img_dir_name, current_label_name)
        label = my_image_open(current_label_path)
        image = my_image_open(current_image_path)
        return transform(image), transform(label)



'''
仅测试
'''
# if __name__ == '__main__':
#     # from torch.nn.functional import one_hot
#     data = MyDataset('/Users/muyichun/Downloads/VOC2007/VOCdevkit3/VOC2007')
#     print(data[0][0].shape)
#     print(data[0][1].shape)
#     # out=one_hot(data[0][1].long())
#     # print(out.shape)