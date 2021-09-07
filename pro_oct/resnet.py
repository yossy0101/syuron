import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv_BN, Conv_BN_ReLU, Oct_AvgPool2d
from octv2_v3 import Oct_BN, Oct_BN_ReLU

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, in_alpha=0.0, out_alpha=0.0, base_width=64, output=False):
        super(BasicBlock, self).__init__()
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        self.stride = stride
        self.output = output
        # input channels
        self.ch_in_lf = int(in_alpha * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        # output channels
        self.ch_out_lf = int(out_alpha * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf
        self.out_alpha = 0.0 if output else out_alpha
        # convlayer
        self.conv1 = Oct_BN_ReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, in_alpha=in_alpha, out_alpha=out_alpha)
        self.conv2 = Oct_BN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, in_alpha=in_alpha, out_alpha=self.out_alpha)
        self.downsample_h = None if in_alpha==1.0 else Conv_BN(self.ch_in_hf, self.ch_out_hf*self.expansion, kernel_size=1, stride=1, padding=0)
        self.downsample_l = None if in_alpha==0.0 else Conv_BN(self.ch_in_lf, self.ch_out_lf*self.expansion, kernel_size=1, stride=1, padding=0)
        self.out_sc = Oct_BN(in_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, in_alpha=in_alpha, out_alpha=self.out_alpha)
        self.avgpool = Oct_AvgPool2d(in_channels, in_alpha)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        sc_h = x[0] if type(x) is tuple else x
        sc_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        if self.stride==2:
            sc_h, sc_l = self.avgpool((sc_h, sc_l))
            sc_h = self.downsample_h(sc_h) if sc_h is not None else None
            sc_l = self.downsample_l(sc_l) if sc_l is not None else None
        if self.output:
            sc_h, sc_l = self.out_sc((sc_h, sc_l))

        x_h = x_h + sc_h if sc_h is not None else x_h
        x_l = x_l + sc_l if sc_l is not None else x_l

        x_h = self.relu(x_h) if x_h is not None else None
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, in_alpha=0.0, out_alpha=0.0, base_width=64, output=False):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (base_width / 64.))
        self.stride = stride
        self.output = output
        # input channels
        self.ch_in_lf = int(in_alpha * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        # output channels
        self.ch_out_lf = int(out_alpha * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf
        self.out_alpha = 0.0 if self.output else out_alpha
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Oct_BN_ReLU(in_channels, width, kernel_size=1, stride=1, padding=0, in_alpha=in_alpha, out_alpha=out_alpha)
        self.conv2 = Oct_BN_ReLU(width, width, kernel_size=3, stride=stride, padding=1,  in_alpha=in_alpha, out_alpha=out_alpha)
        self.conv3 = Oct_BN(width, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, in_alpha=in_alpha, out_alpha=self.out_alpha)
        self.downsample_h = None if in_alpha==1.0 else Conv_BN(self.ch_in_hf, self.ch_out_hf*self.expansion, kernel_size=1, stride=1, padding=0)
        self.downsample_l = None if in_alpha==0.0 else Conv_BN(self.ch_in_lf, self.ch_out_lf*self.expansion, kernel_size=1, stride=1, padding=0)
        self.out_sc = Oct_BN(in_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, in_alpha=in_alpha, out_alpha=self.out_alpha)
        self.avgpool = Oct_AvgPool2d(in_channels, in_alpha)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        sc_h = x[0] if type(x) is tuple else x
        sc_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.stride==2:
            sc_h, sc_l = self.avgpool((sc_h, sc_l))
            sc_h = self.downsample_h(sc_h) if sc_h is not None else None
            sc_l = self.downsample_l(sc_l) if sc_l is not None else None
        if self.output:
            sc_h, sc_l = self.out_sc((sc_h, sc_l))

        x_h = x_h + sc_h if sc_h is not None else x_h
        x_l = x_l + sc_l if sc_l is not None else x_l

        x_h = self.relu(x_h) if x_h is not None else None
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, alpha=0.0, width_per_group=64):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.alpha = alpha
        self.base_width = width_per_group
        self.conv1 = Conv_BN_ReLU(3, self.in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Oct_BN_ReLU(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, in_alpha=0.0, out_alpha=self.alpha)
        self.conv3 = Oct_BN_ReLU(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, in_alpha=self.alpha, out_alpha=self.alpha)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, output=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, output=False):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, in_alpha=self.alpha, out_alpha=self.alpha, base_width=self.base_width))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            out_put=True if (i==blocks-1) and (output==True) else False
            layers.append(block(self.in_channels, out_channels, stride=1, in_alpha=self.alpha, out_alpha=self.alpha, base_width=self.base_width, output=out_put))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x_h, x_l = self.conv2((x, None))
        x_h, x_l = self.conv3((x_h, x_l))
        
        x_h, x_l = self.layer1((x_h, x_l))
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))
        x_h, x_l = self.layer4((x_h, x_l))

        x = self.avgpool(x_h) if x_h is not None else self.avgpool(x_l)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=True)
        x = self.fc(x)

        return x

def _resnet(block, layers, num_classes, alpha, **kwargs):
    model = ResNet(block, layers, num_classes, alpha, **kwargs)
    return model