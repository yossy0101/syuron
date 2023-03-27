import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv_BN_ReLU, fc_ReLU_Drop
from pytorch_memlab import profile
from pytorch_memlab import MemReporter

class VGG(nn.Module):
    def __init__(self, layers, num_classes=100):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        self.layer5 = self._make_layer(512, layers[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = fc_ReLU_Drop(self.in_channels, 4096)
        self.fc2 = fc_ReLU_Drop(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def _make_layer(self, out_channels, blocks):
        layers = []
        layers.append(Conv_BN_ReLU(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(Conv_BN_ReLU(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=True)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
def _vgg(layers, num_classes, **kwargs):
    model = VGG(layers, num_classes, **kwargs)
    return model