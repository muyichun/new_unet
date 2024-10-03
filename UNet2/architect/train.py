import os

import tqdm
from torch import optim
from torch.utils.data import DataLoader

from UNet3.architect.net import UNet
from data import *
from xxxxnet import *
from torchvision.utils import save_image

'''
    初始参数配置
'''
weight_name = "unet.pth"
weight_path = r'/Users/muyichun/PycharmProjects/new_unet/UNet3/'
train_img = 'exp_128'
train_label = 'label_128'
valid_img = 'exp_valid_128'
valid_label = 'label_valid_128'
data_path = r'/Users/muyichun/Desktop/Demo/MCF_speckle_digits/'
save_path = r'/Users/muyichun/PycharmProjects/new_unet/UNet3/train_image/'
batch_size = 1
epochs = 200


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
    程序入口
'''
if __name__ == '__main__':
    train_data_loader = DataLoader(MyDataset(data_path, train_img, train_label), batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(MyDataset(data_path, valid_img, valid_label), batch_size=batch_size, shuffle=True)
    net = UNet().to(device) #GPU
    if os.path.exists(weight_path + weight_name):
        net.load_state_dict(torch.load(weight_path + weight_name))
        print('### successful load weight！')
    else:
        print('### not successful load weight')

    opt = optim.Adam(net.parameters(), lr=1e-4)
    loss_fun = nn.BCELoss()

    best_score = 999
    for epoch in range(epochs):
        # 训练
        net.train()
        for i, (image, label) in enumerate(tqdm.tqdm(train_data_loader)):
            image, label = image.to(device), label.to(device) #GPU
            out_image = net(image)
            train_loss = loss_fun(out_image, label)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch+1}-{i}-train_loss===>>{train_loss.item()}')
            img = torch.stack([image[0], label[0], out_image[0]], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        # 验证
        net.eval()
        with torch.no_grad():
            total_loss = 0.0
            for image, label in valid_data_loader:
                image, label = image.to(device), label.to(device)  # GPU
                out_image = net(image)
                loss = loss_fun(out_image, label)
                total_loss += loss.item()
            average_loss = total_loss / len(valid_data_loader)

            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {average_loss:.4f}")

        # 保存性能最好的模型
        if best_score > average_loss:
            best_score = average_loss
            torch.save(net.state_dict(), weight_path + str(round(best_score,4)) + weight_name)