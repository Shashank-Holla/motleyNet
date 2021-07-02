import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, normType, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.norm1 = self.norm_layer(normType, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.norm2 = self.norm_layer(normType, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def norm_layer(self, norm_type, channels):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'GroupNorm':
            return nn.GroupNorm(num_groups=int(channels/2), num_channels=channels)
        elif norm_type == 'LayerNorm':
            return nn.GroupNorm(num_groups=1, num_channels=channels)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, normType, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.normType = normType

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = self.norm_layer(normType, 64)
        self.layer1 = self._make_layer(block, normType, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, normType, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, normType, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, normType, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, normType, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, normType, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def norm_layer(self, norm_type, channels):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'GroupNorm':
            return nn.GroupNorm(num_groups=int(channels/2), num_channels=channels)
        elif norm_type == 'LayerNorm':
            return nn.GroupNorm(num_groups=1, num_channels=channels)

    def forward(self, x):
        out = F.relu(self.norm(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(normType):
    '''
    Argument
        normType : Normalization type. It needs to be one of- 'BatchNorm', 'GroupNorm', 'LayerNorm'
    '''
    assert normType in ('BatchNorm', 'GroupNorm', 'LayerNorm'), "Incorrect normalization applied"
    return ResNet(BasicBlock, [2, 2, 2, 2], normType)


def ResNet34(normType):
    assert normType in ('BatchNorm', 'GroupNorm', 'LayerNorm'), "Incorrect normalization applied"
    return ResNet(BasicBlock, [3, 4, 6, 3], normType)