import time
import torch
import torch.nn as nn
import torch.optim as optim
import math

def _train(net=None, loader=None, criterion=None, optimizer=None, device='cpu'):
    train_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device) #for gpu

            #前の勾配の情報が溜まるのを防ぐために勾配をリセット
            optimizer.zero_grad()         

            #学習
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
    #損失関数と認識率
    t_loss = train_loss/(i+1)
    t_acc = (1.*correct/total)*100
    
    return t_loss, t_acc
            
def _val(net=None, loader=None, criterion=None, device='cpu'):
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): #勾配の計算を無効にする
            for ii, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device) #for gpu

                #評価
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
    #損失関数と認識率
    v_loss = val_loss/(ii+1)
    v_acc = (1.*correct/total)*100
    
    return v_loss, v_acc

def train(epochs=1, net=None, loader=None, model_path=None, result_path=None):
    if net==None or loader==None:
        print('net or loader is None')
        return
    
    torch.backends.cudnn.benchmark = True
    
    #for gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # 複数GPU使用宣言
    parallel = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        parallel = True
    
    #重み読み込み
    print('Weight loading...')
    net.load_state_dict(torch.load(model_path))
    
    #履歴
    history = {"val_acc":[], "time":[]}
    
    #define loss function and optimier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)
    
    print('Training start')
    for epoch in range(epochs):
        print('Epoch:%d/%d'%(epoch+1,epochs),end='')
        start_time = time.time()

        ##############-----------1epochの訓練---------------###################
        ##############-----------訓練データ------------------##################
        net.train()
        train_loss, train_acc = _train(net, loader['train'], criterion, optimizer, device)
        with open(result_path+'/train_loss.txt', 'a') as f:
            print(train_loss, file=f)
        with open(result_path+'/train_acc.txt', 'a') as f:
            print(train_acc, file=f)
        
        #重み保存 -- GPU並列化をしているかで保存方法を変更
        torch.save(net.module.state_dict(), model_path) if parallel==True else torch.save(net.state_dict(), model_path)

        ####################--------テストデータ------------####################
        net.eval()
        val_loss, val_acc = _val(net, loader['test'], criterion, device)
        history["val_acc"].append(val_acc)
        with open(result_path+'/val_loss.txt', 'a') as f:
            print(val_loss, file=f)
        with open(result_path+'/val_acc.txt', 'a') as f:
            print(val_acc, file=f)
        
        one_epoch_time = time.time() - start_time
        history["time"].append(one_epoch_time)
        print(' - %.1fs'%(one_epoch_time),' - loss:%.4f'%(train_loss),' - acc:%.4f%%'%(train_acc),
              ' - val_loss:%.4f'%(val_loss),' - val_acc:%.4f%%'%(val_acc),' -lr:%.4f'%(scheduler.get_last_lr()[0]))
        
        scheduler.step() #lr_decay
        
    print('Training finished')
    
    with open(result_path+'/max_acc.txt', 'a') as f:
        print('max_accuracy:%f%%'%max(history["val_acc"]), file=f)
    with open(result_path+'/time.txt', 'a') as f:
        print('avarage time/1epoch:%fs'%(sum(history["time"]) / len(history["time"])), file=f)
            
    return