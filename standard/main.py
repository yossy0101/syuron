import os
import sys
sys.path.append(os.path.abspath("../common"))
from data_load import data_load
from model import make_model
from train import train

IMAGE_SIZE = 256
BATCH_SIZE = 128
EPOCHS = 150

#Main
def main():    
    #データセットを選択，読み込み
    datasets = input("What datasets do you use: ")
    datasets, (loader, num_classes) = data_load(IMAGE_SIZE, BATCH_SIZE, datasets)
    
    #Modelを選択, 生成
    use_model = input('What model do you use: ')
    use_model, net = make_model(use_model, num_classes)
    
    #学習
    train(EPOCHS, net, loader, datasets, use_model)
    
if __name__ == '__main__':
    main()