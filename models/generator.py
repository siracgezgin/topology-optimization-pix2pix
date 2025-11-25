"""
Topoloji Optimizasyonu için U-Net Generator Mimarisi
Pix2Pix makalesine dayalı: https://arxiv.org/abs/1611.07004
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Encoder bloğu: Conv2d -> BatchNorm -> LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """Decoder bloğu: ConvTranspose2d -> BatchNorm -> ReLU -> Dropout"""
    
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        x = self.model(x)
        # Encoder'dan gelen skip connection ile birleştir
        return torch.cat((x, skip_connection), 1)


class GeneratorUNet(nn.Module):
    """
    Skip Connection'lı U-Net Generator
    Girdi: 256x256x3 RGB görüntü
    Çıktı: 256x256x3 RGB görüntü
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder (Örnekleme Azaltma)
        self.down1 = DownBlock(in_channels, 64, normalize=False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize=False)

        # Decoder (Örnekleme Artırma)
        self.up1 = UpBlock(512, 512, dropout=True)
        self.up2 = UpBlock(1024, 512, dropout=True)
        self.up3 = UpBlock(1024, 512, dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        # Son katman
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder ileri geçişi
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder ileri geçişi (skip connection'larla)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
