import torch
from torch import nn
from torch.nn import functional as F

'''尺寸不变-池化层'''
class Max_Pool2d(nn.Module):
    def __init__(self):
        super(Max_Pool2d, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        return self.layer(x)

'''尺寸不变-卷积快'''
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layer(x)

'''下采样，通道*2，尺寸/2'''
class DownSampleConv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampleConv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.layer(x)

'''转置卷积，上采样，放大两倍'''
class TransposedConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransposedConvLayer, self).__init__()
        # 定义转置卷积层
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),  # 或者使用其他激活函数如 nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)

'''上采样，组合连接层'''
class ConcatenationLayer(nn.Module):
    def __init__(self, in_channel):
        super(ConcatenationLayer, self).__init__()
        # self.layer = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1) #--改
    def forward(self, x, feature_map):
        # 分辨率放大2倍
        # up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 通道数减半，为了进行两个通道拼凑
        # out = self.layer(up)
        return torch.cat((x, feature_map), dim=1)

class Dropout2dLayer(nn.Module):
    def __init__(self, p=0.4, inplace=True):
        super(Dropout2dLayer, self).__init__()
        self.dropout2d = nn.Dropout2d(p=p, inplace=inplace)
    def forward(self, x):
        return self.dropout2d(x)

'''
UNet主架构
'''
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # encode
        self.in_layer1_1 = Conv_Block(1,128)
        self.in_layer1_2 = Conv_Block(128, 128)
        self.in_maxPool1 = Max_Pool2d()  # 64^2

        self.in_downSample1 = DownSampleConv_Block(128,256)
        self.in_layer2 = Conv_Block(256, 256)
        self.in_maxPool2 = Max_Pool2d()  # 32^2

        self.in_downSample2 = DownSampleConv_Block(256,512)
        self.in_layer3 = Conv_Block(512, 512)
        self.in_maxPool3 = Max_Pool2d()  # 16^2

        self.in_downSample3 = DownSampleConv_Block(512,1024)
        self.in_layer4 = Conv_Block(1024, 1024)
        self.in_transposedConvLayer1 = TransposedConvLayer(1024, 512)  # 32^2

        # decode
        self.out_concatenationLayer1 = ConcatenationLayer(512)
        self.out_layer1 = Conv_Block(1024,512)
        self.out_layer2 = Conv_Block(512, 512)
        self.out_transposedConvLayer2 = TransposedConvLayer(512, 256)  # 64^2

        self.out_concatenationLayer2 = ConcatenationLayer(256)
        self.out_layer3 = Conv_Block(512,256)
        self.out_layer4 = Conv_Block(256, 256)
        self.out_transposedConvLayer3 = TransposedConvLayer(256, 128)

        self.out_concatenationLayer3 = ConcatenationLayer(128)
        self.out_layer5 = Conv_Block(256,128)
        self.out_layer6 = Conv_Block(128,128)
        self.out_dropout = Dropout2dLayer()
        self.out_layer7 = Conv_Block(128, 64)

        self.out=nn.Conv2d(64,1,3,1,1)
        self.Th = nn.Sigmoid()

    def forward(self,x):
        # encode
        encode11 = self.in_layer1_1(x)
        encode12 = self.in_layer1_2(encode11)
        encode13 = self.in_maxPool1(encode12)

        encode21 = self.in_downSample1(encode13)
        encode22 = self.in_layer2(encode21)
        encode23 = self.in_maxPool2(encode22)

        encode31 = self.in_downSample2(encode23)
        encode32 = self.in_layer3(encode31)
        encode33 = self.in_maxPool3(encode32)

        encode41 = self.in_downSample3(encode33)
        encode42 = self.in_layer4(encode41)
        encode43 = self.in_transposedConvLayer1(encode42)

        #decode
        decode11 = self.out_concatenationLayer1(encode43, encode32)
        decode12 = self.out_layer1(decode11)
        decode13 = self.out_layer2(decode12)
        decode14 = self.out_transposedConvLayer2(decode13)

        decode21 = self.out_concatenationLayer2(decode14, encode22)
        decode22 = self.out_layer3(decode21)
        decode23 = self.out_layer4(decode22)
        decode24 = self.out_transposedConvLayer3(decode23)

        decode31 = self.out_concatenationLayer3(decode24, encode12)
        decode32 = self.out_layer5(decode31)
        decode33 = self.out_layer6(decode32)
        decode34 = self.out_dropout(decode33)
        decode35 = self.out_layer7(decode34)

        return self.Th(self.out(decode35))

if __name__ == '__main__':
    x=torch.randn(1,1,128,128)
    net=UNet()
    print(net(x).shape)