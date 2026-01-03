import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        # ---------- VGG16-BN Frontend ----------
        if load_pretrained:
            vgg = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.IMAGENET1K_V1
            )
        else:
            vgg = models.vgg16_bn(weights=None)

        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:33]
        )

        # ---------- CSRNet Backend ----------
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    # ---------- Proper Weight Initialization ----------
    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
