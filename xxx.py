import torch
import torch.nn as nn

# 假设输入图片的尺寸为 (batch_size, in_channels, height, width)
# in_channels 表示输入图片的通道数
in_channels = 64   # 输入通道数
out_channels = in_channels // 2  # 输出通道数减半
kernel_size = 4  # 卷积核大小
stride = 2  # 步幅，表示放大两倍
padding = 1  # 保证输出尺寸为 height*2, width*2

# 定义转置卷积层
trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

# 创建一个随机输入张量
input_tensor = torch.randn(1, in_channels, 32, 32)  # 假设输入图片尺寸为 32x32

# 通过转置卷积层
output_tensor = trans_conv(input_tensor)

print("输入张量尺寸：", input_tensor.shape)
print("输出张量尺寸：", output_tensor.shape)
