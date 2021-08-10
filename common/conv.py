import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Oct_AvgPool2d(nn.Module):
    def __init__(self, in_channels, in_alpha=0.0):
        super(Oct_AvgPool2d, self).__init__()
        # input channels
        self.ch_in_lf = int(in_alpha * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x_h, x_l = x

        x_h = self.pool(x_h) if x_h is not None else None
        x_l = self.pool(x_l) if x_l is not None else None
        return x_h, x_l

class Oct_MaxPool2d(nn.Module):
    def __init__(self, in_channels, in_alpha=0.0):
        super(Oct_MaxPool2d, self).__init__()
        # input channels
        self.ch_in_lf = int(in_alpha * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x_h, x_l = x
        
        x_h = self.pool(x_h) if x_h is not None else None
        x_l = self.pool(x_l) if x_l is not None else None
        return x_h, x_l

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