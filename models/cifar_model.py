import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilated=False):
        super(BasicBlock, self).__init__()
        self.dilated = dilated
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)            
        )
        if self.dilated:
            self.dilated = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), dilation=2, padding=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)            
            )
        self.antman = nn.Sequential(
            nn.Conv2d(in_channels=(in_channels+out_channels), out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        ) 
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.convblock(x)
        if self.dilated:
            outdilated = self.dilated(x)
            # print("Conv shape", out.shape)
            # channel size = outchannels + inchannels
            out = torch.cat([out, outdilated], dim=1)
            out = self.antman(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_depth_dilated0 = BasicBlock(3, 64, True) 

        # Dilation convolution - 1
        self.dilationblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, stride=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) 

        self.conv_depth_dilated1 = BasicBlock(32, 64)

        # Dilation convolution - 2
        self.dilationblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, stride=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) 

        self.conv_depth_dilated2 = BasicBlock(32, 64)         

        # Dilation convolution - 3
        self.dilationblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) 

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=72, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(72),  
            nn.Conv2d(in_channels=72, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)                 
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        # nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        # x = self.convblock1(x)
        x = self.conv_depth_dilated0(x)
        x = self.dilationblock1(x)
        x = self.conv_depth_dilated1(x)
        x = self.dilationblock2(x)
        x = self.conv_depth_dilated2(x)
        x = self.dilationblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return x