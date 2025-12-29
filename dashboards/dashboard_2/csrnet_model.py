import torch
import torch.nn as nn

cfg = {
    'frontend': [64,64,'M',128,128,'M',256,256,256,'M',512,512,512],
    'backend' : [512,512,512,256,128,64]
}

def make_layers(cfg_list, in_channels=3, dilation=False):
    layers = []
    for v in cfg_list:
        if v == 'M':
            layers.append(nn.MaxPool2d(2,2))
        else:
            d = 2 if dilation else 1
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d),
                nn.ReLU(inplace=True)
            ]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        
        self.frontend = make_layers(cfg['frontend'], in_channels=3, dilation=False)
        self.backend  = make_layers(cfg['backend'], in_channels=512, dilation=True)
        self.output   = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        return x   