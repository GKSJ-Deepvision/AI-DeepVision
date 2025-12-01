import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # FRONTEND (VGG16 layers)
        self.frontend = models.vgg16(pretrained=True).features[:23]

        # BACKEND (Dilated convolutions)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # OUTPUT LAYER — Density Map
        self.output_layer = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
