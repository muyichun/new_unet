from UNet1.architect.net import *
from UNet1.architect.data import *
from torchvision.utils import save_image

'''
    测试目录下所有图片
'''

imgs_path = "/Users/muyichun/Downloads/VOC2007/VOCdevkit3/VOC2007/xxx"
predict_path = "/Users/muyichun/Downloads/VOC2007/VOCdevkit3/VOC2007/predict/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=UNet().to(device)
net.load_state_dict(torch.load('params/unet.pth'))

imgs_dir = os.listdir(imgs_path)
for e in imgs_dir:
    img_path = keep_image_size_open_rgb(os.path.join(imgs_path, e))
    img_tensor_3dim = transform(img_path).to(device)
    img_tensor_4dim = torch.unsqueeze(img_tensor_3dim, dim=0)
    out = net(img_tensor_4dim)
    save_image(out, os.path.join(predict_path, e))