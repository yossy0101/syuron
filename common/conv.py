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
    
class OctConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, in_alpha=0.0, out_alpha=0.0, bias=True):
        super(OctConv2d, self).__init__()
        assert (0 <= in_alpha <= 1) and (0 <= out_alpha <= 1), "Alphas must be in interval [0, 1]"

        # input channels
        self.ch_in_lf = int(in_alpha * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        # output channels
        self.ch_out_lf = int(out_alpha * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf

        # conv layers
        self.HtoH, self.HtoL, self.LtoH, self.LtoL = None, None, None, None
        #HtoH
        if not (self.ch_out_hf == 0 or self.ch_in_hf == 0):
            self.HtoH = nn.Conv2d(self.ch_in_hf, self.ch_out_hf, kernel_size, stride, padding, bias=bias)
        #HtoL
        if not (self.ch_out_lf == 0 or self.ch_in_hf == 0):
            self.HtoL = nn.Conv2d(self.ch_in_hf, self.ch_out_lf, kernel_size, stride, padding, bias=bias)
        #LtoH
        if not (self.ch_out_hf == 0 or self.ch_in_lf == 0):
            self.LtoH = nn.Conv2d(self.ch_in_lf, self.ch_out_hf, kernel_size, stride, padding, bias=bias)
        #LtoL
        if not (self.ch_out_lf == 0 or self.ch_in_lf == 0):
            self.LtoL = nn.Conv2d(self.ch_in_lf, self.ch_out_lf, kernel_size, stride, padding, bias=bias)
    
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            
    def forward(self, x):
        hf, lf = x

        # apply convolutions
        oHtoH = oHtoL = oLtoH = oLtoL = 0.
        if self.HtoH is not None:
            oHtoH = self.HtoH(hf)
        if self.HtoL is not None:
            oHtoL = self.HtoL(self.downsample(hf))
        if self.LtoH is not None:
            oLtoH = self.upsample(self.LtoH(lf))
        if self.LtoL is not None:
            oLtoL = self.LtoL(lf)
        
        # compute output tensors
        hf = oHtoH + oLtoH if self.ch_out_hf != 0 else None
        lf = oLtoL + oHtoL if self.ch_out_lf != 0 else None
        return hf, lf

class Oct_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, in_alpha=0.0, out_alpha=0.0):
        super(Oct_BN, self).__init__()
        self.conv = OctConv2d(in_channels, out_channels, kernel_size, stride, padding, in_alpha, out_alpha)
        self.ch_out_lf = int(out_alpha * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf
        self.bn_h = None if out_alpha == 1.0 else nn.BatchNorm2d(self.ch_out_hf)
        self.bn_l = None if out_alpha == 0.0 else nn.BatchNorm2d(self.ch_out_lf)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h) if x_h is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l

class Oct_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, in_alpha=0.0, out_alpha=0.0):
        super(Oct_BN_ReLU, self).__init__()
        self.conv = OctConv2d(in_channels, out_channels, kernel_size, stride, padding, in_alpha, out_alpha)
        self.ch_out_lf = int(out_alpha * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf
        self.bn_h = None if out_alpha == 1.0 else nn.BatchNorm2d(self.ch_out_hf)
        self.bn_l = None if out_alpha == 0.0 else nn.BatchNorm2d(self.ch_out_lf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h)) if x_h is not None else None
        x_l = self.relu(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l

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