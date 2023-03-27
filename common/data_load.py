import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
sys.path.append(os.path.abspath("/home/csl-stu/share/torch-datasets/tiny_imagenet_200"))
from tiny import TImgNetDataset

#データセット読み込み
def data_load(size=224, batch=32, datasets='cifar100'):
   if datasets=='cifar10':
      return data_load_CIFAR10(size, batch)
   elif datasets=='cifar100':
      return data_load_CIFAR100(size, batch)
   elif datasets=='food101':
      return data_load_Food101(size, batch)
   elif datasets=='tiny':
      return data_load_tinyimagenet(size, batch)
   elif datasets=='imagenet100':
      return data_load_ImageNet100(size, batch)
   elif datasets=='imagenet':
      return data_load_ImageNet(size, batch)

def transform_train(size=224):
   transform = transforms.Compose([
      transforms.Resize((size,size)), 
      transforms.RandomCrop(size, padding=int(size/8)),
      transforms.RandomHorizontalFlip(),                                   
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
   return transform

def transform_test(size=224):
   transform = transforms.Compose([
      transforms.Resize((size,size)),  
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
   return transform

def data_load_CIFAR10(size=224, batch=32): 
   print("loading CIFAR10 dataset...")     
   trainset = CIFAR10(root='/home/csl-stu/share/torch-datasets', train=True,download=True, transform=transform_train(size))
   trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)

   testset = CIFAR10(root='/home/csl-stu/share/torch-datasets', train=False,download=True, transform=transform_test(size))
   testloader = DataLoader(testset, batch_size=batch,shuffle=False, num_workers=2)

   return {'train':trainloader, 'test':testloader}, 10
   
def data_load_CIFAR100(size=224, batch=32):    
   print("loading CIFAR100 dataset...")   
   trainset = CIFAR100(root='/home/csl-stu/share/torch-datasets', train=True,download=True, transform=transform_train(size))
   trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)

   testset = CIFAR100(root='/home/csl-stu/share/torch-datasets', train=False,download=True, transform=transform_test(size))
   testloader = DataLoader(testset, batch_size=batch,shuffle=False, num_workers=2)

   return {'train':trainloader, 'test':testloader}, 100

def data_load_Food101(size=224, batch=32):
   print("loading Food-101 dataset...")    
   trainset = ImageFolder('/home/csl-stu/share/torch-datasets/food-101/train', transform_train(size))
   trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
   
   testset = ImageFolder('/home/csl-stu/share/torch-datasets/food-101/test', transform_test(size))
   testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
   
   return {'train':trainloader, 'test':testloader}, 101

def data_load_tinyimagenet(size=224, batch=32):
   print("loading tiny-imagenet-200 dataset...") 
   trainset = ImageFolder('/home/csl-stu/share/torch-datasets/tiny_imagenet_200/train', transform_train(size))
   trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
   
   valdir = '/home/csl-stu/share/torch-datasets/tiny_imagenet_200/val/images'
   valgtfile = '/home/csl-stu/share/torch-datasets/tiny_imagenet_200/val/val_annotations.txt'
   testset = TImgNetDataset(valdir, valgtfile, trainloader.dataset.class_to_idx.copy(), transform_test(size))
   testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
   
   return {'train':trainloader, 'test':testloader}, 200

def data_load_ImageNet100(size=224, batch=32):
   print("loading ImageNet100 dataset...")    
   trainset = ImageFolder('~/work/torch-dataset/data/imagenet100/train', transform_train(size))
   trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
   
   testset = ImageFolder('~/work/torch-dataset/data/imagenet100/val', transform_test(size))
   testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
   
   return {'train':trainloader, 'test':testloader}, 100

def data_load_ImageNet(size=224, batch=32):
   print("loading ImageNet dataset...")    
   trainset = ImageFolder('~/work/torch-dataset/data/imagenet/train', transform_train(size))
   trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
   
   testset = ImageFolder('~/work/torch-dataset/data/imagenet/val', transform_test(size))
   testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
   
   return {'train':trainloader, 'test':testloader}, 1000