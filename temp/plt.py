import matplotlib.pyplot as plt
import pylab

####################--------グラフを保存------------####################
#認識率
fig1 = plt.figure()
plt.plot(history['acc'], label="acc")
plt.plot(history['val_acc'], label="val_acc")
plt.xlabel("epoch")
plt.ylabel("accracy")
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
pylab.subplots_adjust(right=0.7)
fig1.savefig(path+"/acc.png")
    
#損失関数
fig2 = plt.figure()
plt.plot(history['loss'], label="loss")
plt.plot(history['val_loss'], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
pylab.subplots_adjust(right=0.7)
fig2.savefig(path+"/loss.png")