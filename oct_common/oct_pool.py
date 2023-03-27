import torch
import torch.nn as nn

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