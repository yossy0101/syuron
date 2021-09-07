import argparse
import os
import sys
sys.path.append(os.path.abspath("../common"))
from data_load import data_load
from model import make_model
from train import train

parser = argparse.ArgumentParser(description='PyTorch Proposal OctConv')
parser.add_argument('-image_size', default=256, type=int)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-epochs', default=150, type=int)
parser.add_argument('-alpha', default=0.25, type=float)
parser.add_argument('-datasets', help='choices detasets', default='cifar100', 
                    choices=['cifar10', 'cifar100', 'food101', 'tiny'])
parser.add_argument('-use_model', help='choices model', default='resnet18', 
                    choices=['vgg16', 'vgg19', 'resent18', 'resnet34', 'resnet50', 'resnet101'])

#Main
def main():   
    args = parser.parse_args()
    #データセットを選択，読み込み
    loader, num_classes = data_load(args.image_size, args.batch_size, args.datasets)
    
    #Modelを選択, 生成
    net = make_model(args.use_model, num_classes, args.alpha)
    
    #重みの保存先
    model_path = './weight/' + args.use_model + '/octv2_v3/model.pth'

    #結果の保存先
    result_path = './result/' + args.use_model + '/octv2_v3'

    #学習
    train(args.epochs, net, loader, model_path, result_path)
    
if __name__ == '__main__':
    main()