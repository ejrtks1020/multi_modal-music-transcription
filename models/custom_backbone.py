import torch
import torch.nn as nn


def conv_block(
               in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               pool=False,
               norm=False):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 
                  out_channels=out_channels, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding)
    )
    if norm:
        block.append(nn.BatchNorm2d(out_channels))
    if pool:
        block.append(nn.MaxPool2d(2, 2))
    return block

class CustomBackbone(nn.Module):
    def __init__(self, output_size):
        super(CustomBackbone, self).__init__()
        self.backbone = nn.Sequential(
            conv_block(5, 32, 3, 1, 1, norm=True),
            conv_block(32, 64, 3, 1, 1, pool=True, norm=True),
            conv_block(64, 64, 3, 1, 1, pool=True, norm=True),
            conv_block(64, 128, 3, 1, 1, pool=True),
            conv_block(128, 256, 3, 1, 1),
            nn.MaxPool2d((1, 2), (1,2)),            
            conv_block(256, 256, 3, 1, 1),
            nn.MaxPool2d((1, 2), (1,2)),
            conv_block(256, 512, 3, 1, 1),
            nn.MaxPool2d((1, 2), (1,2)),
            conv_block(512, 512, 3, 1, 1),
            nn.MaxPool2d((1, 3), (1,1)),
            conv_block(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x.squeeze())
        return out
    
if __name__ == "__main__":
    net = CustomBackbone(64)
    x = torch.randn(2, 5, 100, 900)
    print(net(x).shape)