import torch
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filter_num=128, filter_size=3):
        super(UNet, self).__init__()


        # 下采样：1
        self.conv1_1 = nn.Conv2d(in_channels, filter_num, kernel_size=filter_size, padding='same')
        self.conv1_2 = nn.Conv2d(filter_num, filter_num, kernel_size=filter_size, padding='same')
        self.encoder_stage_1 = nn.Sequential(
            self.conv1_1,
            nn.ReLU(),
            self.conv1_2,
            nn.ReLU()
        )

        # 下采样：2
        self.conv1_3 = nn.Conv2d(filter_num, filter_num * 2, kernel_size=filter_size, padding='same')
        self.conv1_4 = nn.Conv2d(filter_num * 2, filter_num * 2, kernel_size=filter_size, padding='same')
        self.encoder_stage_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv1_3,
            nn.ReLU(),
            self.conv1_4,
            nn.ReLU()
        )

        # 下采样：3
        self.conv1_5 = nn.Conv2d(filter_num * 2, filter_num * 4, kernel_size=filter_size, padding='same')
        self.conv1_6 = nn.Conv2d(filter_num * 4, filter_num * 4, kernel_size=filter_size, padding='same')
        self.encoder_stage_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv1_5,
            nn.ReLU(),
            self.conv1_6,
            nn.ReLU()
        )

        # 底部层
        self.conv1_7 = nn.Conv2d(filter_num * 4, filter_num * 8, kernel_size=filter_size, padding='same')
        self.conv1_8 = nn.Conv2d(filter_num * 8, filter_num * 8, kernel_size=filter_size, padding='same')
        self.bridge = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv1_7,
            nn.ReLU(),
            self.conv1_8,
            nn.ReLU(),
        )

        # 解码器部分
        self.decoder_stage_1_upconv = nn.ConvTranspose2d(filter_num * 8, filter_num * 4, kernel_size=(2,2), stride=(2,2))
        nn.init.kaiming_normal_(self.decoder_stage_1_upconv.weight, mode='fan_in', nonlinearity='relu')
        # --cat
        self.conv2_1 = nn.Conv2d(filter_num * 8, filter_num * 4, kernel_size=filter_size, padding='same')
        self.conv2_2 = nn.Conv2d(filter_num * 4, filter_num * 4, kernel_size=filter_size, padding='same')
        self.decoder_stage_1 = nn.Sequential(
            self.conv2_1,
            nn.ReLU(),
            self.conv2_2,
            nn.ReLU()
        )

        self.decoder_stage_2_upconv = nn.ConvTranspose2d(filter_num * 4, filter_num * 2, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.decoder_stage_2_upconv.weight, mode='fan_in', nonlinearity='relu')
        # --cat
        self.conv2_3 = nn.Conv2d(filter_num * 4, filter_num * 2, kernel_size=filter_size, padding='same')
        self.conv2_4 = nn.Conv2d(filter_num * 2, filter_num * 2, kernel_size=filter_size, padding='same')
        self.decoder_stage_2 = nn.Sequential(
            self.conv2_3,
            nn.ReLU(),
            self.conv2_4,
            nn.ReLU()
        )

        self.decoder_stage_3_upconv = nn.ConvTranspose2d(filter_num * 2, filter_num, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.decoder_stage_2_upconv.weight, mode='fan_in', nonlinearity='relu')
        # --cat
        self.conv2_5 = nn.Conv2d(filter_num * 2, filter_num, kernel_size=filter_size, padding='same')
        self.conv2_6 = nn.Conv2d(filter_num, filter_num, kernel_size=filter_size, padding='same')
        self.conv2_7 = nn.Conv2d(filter_num, filter_num, kernel_size=filter_size, padding='same')
        self.conv2_8 = nn.Conv2d(filter_num, filter_num, kernel_size=filter_size, padding='same')
        self.decoder_stage_3 = nn.Sequential(
            self.conv2_5,
            nn.ReLU(),
            self.conv2_6,
            nn.ReLU()
        )
        self.decoder_stage_4 = nn.Sequential(
            nn.Dropout(p=0.5),
            self.conv2_7,
            nn.ReLU(),
            self.conv2_8,
            nn.ReLU(),
            nn.Conv2d(filter_num, out_channels, kernel_size=1)
        )
        # 输出
        self.Th = nn.Sigmoid()

    def forward(self, x):
        # 编码器
        enc1 = self.encoder_stage_1(x)
        enc2 = self.encoder_stage_2(enc1)
        enc3 = self.encoder_stage_3(enc2)

        bridge = self.bridge(enc3)

        # 解码器
        dec1_up = self.decoder_stage_1_upconv(bridge)
        dec1 = self.decoder_stage_1(torch.cat([dec1_up, enc3], dim=1))
        dec2_up = self.decoder_stage_2_upconv(dec1)
        dec2 = self.decoder_stage_2(torch.cat([dec2_up, enc2], dim=1))
        dec3_up = self.decoder_stage_3_upconv(dec2)
        dec3 = self.decoder_stage_3(torch.cat([dec3_up, enc1], dim=1))
        output = self.decoder_stage_4(dec3)

        return self.Th(output)
        # return output


if __name__ == '__main__':
    x=torch.randn(1,1,128,128)
    net=UNet()
    print(net(x).shape)