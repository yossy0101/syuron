from vgg import _vgg
from resnet import _resnet, BasicBlock, Bottleneck

def make_model(model='resnet18', num_classes=100, alpha=0.0):
    if model=='vgg16':
        return vgg16(num_classes, alpha)
    elif model=='vgg19':
        return vgg19(num_classes, alpha)
    elif model=='resnet18':
        return resnet18(num_classes, alpha)
    elif model=='resnet34':
        return resnet34(num_classes, alpha)
    elif model=='resnet50':
        return resnet50(num_classes, alpha)
    elif model=='resnet101':
        return resnet101(num_classes, alpha)
    
def vgg16(num_classes=100, alpha=0.0, **kwargs):
    return _vgg([2, 2, 3, 3, 3], num_classes, alpha, **kwargs)

def vgg19(num_classes=100, alpha=0.0, **kwargs):
    return _vgg([2, 2, 4, 4, 4], num_classes, alpha, **kwargs)

def resnet18(num_classes=100, alpha=0.0, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, alpha, **kwargs)

def resnet34(num_classes=100, alpha=0.0, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, alpha, **kwargs)

def resnet50(num_classes=100, alpha=0.0, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, alpha, **kwargs)

def resnet101(num_classes=100, alpha=0.0, **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, alpha, **kwargs)