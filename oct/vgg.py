import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Oct_BN_ReLU, Oct_MaxPool2d, fc_ReLU_Drop

class VGG(nn.Module):
    def __init__(self, layers, num_classes=100, alpha=0.0):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.alpha = alpha
        self.layer1 = self._make_layer(64, layers[0], input=True)
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        self.layer5 = self._make_layer(512, layers[4], output=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = fc_ReLU_Drop(self.in_channels, 4096)
        self.fc2 = fc_ReLU_Drop(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def _make_layer(self, out_channels, blocks, input=False, output=False):
        layers = []
        in_alpha = 0.0 if input==True else self.alpha
        layers.append(Oct_BN_ReLU(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1, in_alpha=in_alpha, out_alpha=self.alpha))
        self.in_channels = out_channels
        for i in range(1, blocks):
            out_alpha = 0.0 if (i==blocks-1) and (output==True) else self.alpha
            layers.append(Oct_BN_ReLU(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1, in_alpha=self.alpha, out_alpha=out_alpha))
        layers.append(Oct_MaxPool2d(out_channels, self.alpha))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x_h, x_l = self.layer1((x, None))
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))
        x_h, x_l = self.layer4((x_h, x_l))
        x_h, x_l = self.layer5((x_h, x_l))
        
        x = self.avgpool(x_h) if x_h is not None else self.avgpool(x_l) 
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=True)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
def _vgg(layers, num_classes, alpha, **kwargs):
    model = VGG(layers, num_classes, alpha, **kwargs)
    return model