import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Fixed: in_ch = upsampled channels + skip channels
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Crop skip to match upsampled x dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            skip = skip[:, :, diff_h//2:skip.size(2)-diff_h//2+diff_h%2, 
                           diff_w//2:skip.size(3)-diff_w//2+diff_w%2]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet_regression(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base_ch)           # 32
        self.down2 = DoubleConv(base_ch, base_ch * 2)     # 64  
        self.down3 = DoubleConv(base_ch * 2, base_ch * 4) # 128

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)  # 256

        # Fixed: UpBlock(in_upsampled_ch, skip_ch, out_ch)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)    # (256+128)->128
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)    # (128+64)->64
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)            # (64+32)->32

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)    # 32ch
        c2 = self.down2(self.pool(c1))  # 64ch
        c3 = self.down3(self.pool(c2))  # 128ch

        b = self.bottleneck(self.pool(c3))  # 256ch

        u3 = self.up3(b, c3)    # 128ch
        u2 = self.up2(u3, c2)   # 64ch
        u1 = self.up1(u2, c1)   # 32ch

        out = self.out_conv(u1)
        return out
