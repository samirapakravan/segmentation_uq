from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 1,
                 bilinear: bool =False,
                 ddims: List = [64, 128, 256, 512, 1024],
                 UQ: bool =True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        udims = ddims[::-1]
        
        self.dropout = nn.Dropout(p=0.5 if UQ else 0.0)
        
        self.inc = (DoubleConv(n_channels, ddims[0]))
        self.down1 = (Down(ddims[0], ddims[1]))
        self.down2 = (Down(ddims[1], ddims[2]))
        self.down3 = (Down(ddims[2], ddims[3]))
        self.down4 = (Down(ddims[3], ddims[4] // factor))

        self.up1 = (Up(udims[0], udims[1] // factor, bilinear))
        self.up2 = (Up(udims[1], udims[2] // factor, bilinear))
        self.up3 = (Up(udims[2], udims[3] // factor, bilinear))
        self.up4 = (Up(udims[3], udims[4], bilinear))
        self.outc = (OutConv(udims[4], n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.dropout(self.down1(x1))
        x3 = self.dropout(self.down2(x2))
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout(self.down4(x4))
        x = self.dropout(self.up1(x5, x4))
        x = self.dropout(self.up2(x, x3))
        x = self.dropout(self.up3(x, x2))
        x = self.dropout(self.up4(x, x1))
        logits = self.outc(x)
        return logits

