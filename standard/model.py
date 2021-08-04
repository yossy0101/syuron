from vgg import _vgg
from resnet import _resnet, BasicBlock, Bottleneck

def make_model(model='resnet18', num_classes=100):
    if model=='vgg16':
        return model, vgg16(num_classes)
    elif model=='vgg19':
        return model, vgg19(num_classes)
    elif model=='resnet18':
        return model, resnet18(num_classes)
    elif model=='resnet34':
        return model, resnet34(num_classes)
    elif model=='resnet50':
        return model, resnet50(num_classes)
    elif model=='resnet101':
        return model, resnet101(num_classes)
    else:
        print('That model is not')
        return make_model(input('What model do you use: '))
    
def vgg16(num_classes=100, **kwargs):
    return _vgg([2, 2, 3, 3, 3], num_classes, **kwargs)

def vgg19(num_classes=100, **kwargs):
    return _vgg([2, 2, 4, 4, 4], num_classes, **kwargs)

def resnet18(num_classes=100, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)

def resnet34(num_classes=100, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

def resnet50(num_classes=100, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)

def resnet101(num_classes=100, **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)