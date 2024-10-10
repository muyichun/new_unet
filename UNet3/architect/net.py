import torch
import torch.nn as nn

# Define the network architecture
class UNet(nn.Module):
    def __init__(self, filter_size=3, filter_num=128):
        super(UNet, self).__init__()

        # Encoder Stage 2
        self.encoder_stage_2_conv1 = nn.Conv2d(in_channels=1, out_channels=filter_num, kernel_size=filter_size, padding='same')
        self.encoder_stage_2_relu1 = nn.ReLU()
        self.encoder_stage_2_conv2 = nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=filter_size, padding='same')
        self.encoder_stage_2_relu2 = nn.ReLU()
        self.encoder_stage_2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Stage 3
        self.encoder_stage_3_conv1 = nn.Conv2d(in_channels=filter_num, out_channels=filter_num*2, kernel_size=filter_size, padding='same')
        self.encoder_stage_3_relu1 = nn.ReLU()
        self.encoder_stage_3_conv2 = nn.Conv2d(in_channels=filter_num*2, out_channels=filter_num*2, kernel_size=filter_size, padding='same')
        self.encoder_stage_3_relu2 = nn.ReLU()
        self.encoder_stage_3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Stage 4
        self.encoder_stage_4_conv1 = nn.Conv2d(in_channels=filter_num*4, out_channels=filter_num*8, kernel_size=filter_size, padding='same')
        self.encoder_stage_4_relu1 = nn.ReLU()
        self.encoder_stage_4_conv2 = nn.Conv2d(in_channels=filter_num*8, out_channels=filter_num*8, kernel_size=filter_size, padding='same')
        self.encoder_stage_4_relu2 = nn.ReLU()
        self.encoder_stage_4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge_conv1 = nn.Conv2d(in_channels=filter_num*8, out_channels=filter_num*16, kernel_size=filter_size, padding='same')
        self.bridge_relu1 = nn.ReLU()
        self.bridge_conv2 = nn.Conv2d(in_channels=filter_num*16, out_channels=filter_num*16, kernel_size=filter_size, padding='same')
        self.bridge_relu2 = nn.ReLU()
        self.bridge_upconv = nn.ConvTranspose2d(in_channels=filter_num*16, out_channels=filter_num*8, kernel_size=2, stride=2)
        self.bridge_uprelu = nn.ReLU()

        # Decoder Stage 1
        self.decoder_stage_1_conv1 = nn.Conv2d(in_channels=filter_num*8 * 2, out_channels=filter_num*8, kernel_size=filter_size, padding='same')
        self.decoder_stage_1_relu1 = nn.ReLU()
        self.decoder_stage_1_conv2 = nn.Conv2d(in_channels=filter_num*8, out_channels=filter_num*8, kernel_size=filter_size, padding='same')
        self.decoder_stage_1_relu2 = nn.ReLU()
        self.decoder_stage_1_upconv = nn.ConvTranspose2d(in_channels=filter_num*8, out_channels=filter_num*4, kernel_size=2, stride=2)
        self.decoder_stage_1_uprelu = nn.ReLU()

        # Decoder Stage 2
        self.decoder_stage_2_concat = nn.Identity()
        self.decoder_stage_2_conv1 = nn.Conv2d(in_channels=filter_num*4 * 2, out_channels=filter_num*4, kernel_size=filter_size, padding='same')
        self.decoder_stage_2_relu1 = nn.ReLU()
        self.decoder_stage_2_conv2 = nn.Conv2d(in_channels=filter_num*4, out_channels=filter_num*4, kernel_size=filter_size, padding='same')
        self.decoder_stage_2_relu2 = nn.ReLU()
        self.decoder_stage_2_upconv = nn.ConvTranspose2d(in_channels=filter_num*4, out_channels=filter_num*2, kernel_size=2, stride=2)
        self.decoder_stage_2_uprelu = nn.ReLU()

        # Decoder Stage 3
        self.decoder_stage_3_concat = nn.Identity()
        self.decoder_stage_3_conv1 = nn.Conv2d(in_channels=filter_num*2 * 2, out_channels=filter_num*2, kernel_size=filter_size, padding='same')
        self.decoder_stage_3_relu1 = nn.ReLU()
        self.decoder_stage_3_conv2 = nn.Conv2d(in_channels=filter_num*2, out_channels=filter_num*2, kernel_size=filter_size, padding='same')
        self.decoder_stage_3_relu2 = nn.ReLU()
        self.dropout = nn.Dropout()

        # Decoder Stage 4
        self.decoder_stage_4_conv1 = nn.Conv2d(in_channels=filter_num*2, out_channels=filter_num, kernel_size=filter_size, padding='same')
        self.decoder_stage_4_relu1 = nn.ReLU()
        self.decoder_stage_4_conv2 = nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=filter_size, padding='same')
        self.decoder_stage_4_relu2 = nn.ReLU()
        self.final_conv = nn.Conv2d(in_channels=filter_num, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        enc2 = self.encoder_stage_2_relu1(self.encoder_stage_2_conv1(x))
        enc2 = self.encoder_stage_2_relu2(self.encoder_stage_2_conv2(enc2))
        enc2_pool = self.encoder_stage_2_maxpool(enc2)

        enc3 = self.encoder_stage_3_relu1(self.encoder_stage_3_conv1(enc2_pool))
        enc3 = self.encoder_stage_3_relu2(self.encoder_stage_3_conv2(enc3))
        enc3_pool = self.encoder_stage_3_maxpool(enc3)

        enc4 = self.encoder_stage_4_relu1(self.encoder_stage_4_conv1(enc3_pool))
        enc4 = self.encoder_stage_4_relu2(self.encoder_stage_4_conv2(enc4))
        enc4_pool = self.encoder_stage_4_maxpool(enc4)

        # Bridge
        bridge = self.bridge_relu1(self.bridge_conv1(enc4_pool))
        bridge = self.bridge_relu2(self.bridge_conv2(bridge))
        bridge_up = self.bridge_uprelu(self.bridge_upconv(bridge))

        # Decoder Path
        dec1 = torch.cat((bridge_up, enc4), dim=1)  # Depth concatenation
        dec1 = self.decoder_stage_1_relu1(self.decoder_stage_1_conv1(dec1))
        dec1 = self.decoder_stage_1_relu2(self.decoder_stage_1_conv2(dec1))
        dec1_up = self.decoder_stage_1_uprelu(self.decoder_stage_1_upconv(dec1))

        dec2 = torch.cat((dec1_up, enc3), dim=1)
        dec2 = self.decoder_stage_2_relu1(self.decoder_stage_2_conv1(dec2))
        dec2 = self.decoder_stage_2_relu2(self.decoder_stage_2_conv2(dec2))
        dec2_up = self.decoder_stage_2_uprelu(self.decoder_stage_2_upconv(dec2))

        dec3 = torch.cat((dec2_up, enc2), dim=1)
        dec3 = self.decoder_stage_3_relu1(self.decoder_stage_3_conv1(dec3))
        dec3 = self.decoder_stage_3_relu2(self.decoder_stage_3_conv2(dec3))
        dec3 = self.dropout(dec3)

        dec4 = self.decoder_stage_4_relu1(self.decoder_stage_4_conv1(dec3))
        dec4 = self.decoder_stage_4_relu2(self.decoder_stage_4_conv2(dec4))

        # Final Convolution Layer
        final_output = self.final_conv(dec4)

        # return nn.Sigmoid()(final_output)
        return final_output

if __name__ == '__main__':
    x=torch.randn(1,1,64,64)
    net=UNet()
    print(net(x).shape)