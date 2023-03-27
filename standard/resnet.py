import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv_BN, Conv_BN_ReLU
from pytorch_memlab import profile

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv_BN_ReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv_BN(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_BN(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.stride = stride

    @profile
    def forward(self, x):
        sc = x

        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.stride==2:
            sc = self.downsample(sc)

        x += sc
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv_BN_ReLU(in_channels, width, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv_BN_ReLU(width, width, kernel_size=3, stride=stride, padding=1)
        self.conv3 = Conv_BN(width, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_BN(in_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        )
        self.stride = stride

    def forward(self, x):
        sc = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.stride==2:
            sc = self.downsample(sc)

        x += sc
        x = self.relu(x)

        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, width_per_group=64):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.base_width = width_per_group
        self.conv1 = Conv_BN_ReLU(3, self.in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_BN_ReLU(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, base_width=self.base_width))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, base_width=self.base_width))

        return nn.Sequential(*layers)

    # @profile
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=True)
        x = self.fc(x)

        return x

def _resnet(block, layers, num_classes, **kwargs):
    model = ResNet(block, layers, num_classes, **kwargs)
    return model