import os

import tqdm
from torch import optim
from torch.utils.data import DataLoader
from UNet1.architect.data import *
from UNet1.architect.net import *
from torchvision.utils import save_image

'''
    初始参数配置
'''
weight_path = '/Users/muyichun/PycharmProjects/new_unet/UNet1/params/unet.pth'
img_name = 'exp_128'
label_name = 'label_128'
data_path = r'/Users/muyichun/Desktop/Demo/MCF_speckle_digits/'
save_path = '/Users/muyichun/PycharmProjects/new_unet/UNet1/train_image'
epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
    程序入口
'''
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path, img_name, label_name), batch_size=5, shuffle=True)
    net = UNet().to(device) #GPU
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('### successful load weight！')
    else:
        print('### not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()
    # loss_fun = nn.CrossEntropyLoss()    多分类任务中


    for epoch in range(epochs):
        for i, (image, label) in enumerate(tqdm.tqdm(data_loader)):
            image, label = image.to(device), label.to(device) #GPU
            out_image = net(image)
            train_loss = loss_fun(out_image, label)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch+1}-{i}-train_loss===>>{train_loss.item()}')

            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            img = torch.stack([image[0], label[0], out_image[0]], dim=0)
            save_image(img, f'{save_path}/{i}.png')