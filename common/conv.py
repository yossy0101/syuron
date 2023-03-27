import torch
import torch.nn as nn
from pytorch_memlab import profile
from pytorch_memlab import profile

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class fc_ReLU_Drop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fc_ReLU_Drop, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        return x