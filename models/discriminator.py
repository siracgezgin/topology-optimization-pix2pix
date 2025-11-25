"""
Topoloji Optimizasyonu için PatchGAN Discriminator
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Discriminator bloğu: Conv2d -> BatchNorm -> LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    70x70 örtüşen görüntü parçalarının gerçek veya sahte olduğunu sınıflandırır
    """
    
    def __init__(self, in_channels=6):
        super().__init__()
        self.model = nn.Sequential(
            DownBlock(in_channels, 64, normalize=False),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, img_A, img_B):
        # Girdi ve hedef görüntüleri kanal boyutunda birleştir
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
