import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

def data_load(size=224, batch=32, datasets='cifar100'):
   if datasets=='cifar10':
      return data_load_CIFAR10(size, batch)
  #  elif datasets=='cifar100':
  #     return data_load_CIFAR100(size, batch)
  #  elif datasets=='food101':
  #     return data_load_Food101(size, batch)
  #  elif datasets=='tiny':
  #     return data_load_tinyimagenet(size, batch)
  #  elif datasets=='imagenet100':
  #     return data_load_ImageNet100(size, batch)
  #  elif datasets=='imagenet':
  #     return data_load_ImageNet(size, batch)
    
def data_load_CIFAR10(size=224, batch=32):
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  X_train =[]
  X_test = []
  for i in range(50000):
      dst = cv2.resize(x_train[i], (size, size), interpolation=cv2.INTER_CUBIC)     
      X_train.append(dst)
  for i in range(10000):
      dst = cv2.resize(x_test[i], (size, size), interpolation=cv2.INTER_CUBIC)
      X_test.append(dst)
  X_train = np.array(X_train)
  X_test = np.array(X_test)

  #上記のfor文の個数と以下のコードで学習するデータの個数と検証用データの個数を調整
  y_train=y_train[:50000]
  y_test=y_test[:10000]

  X_train = X_train.astype('float32')/255.0
  X_test = X_test.astype('float32')/255.0
  
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)
    
  train_gen = ImageDataGenerator()#width_shift_range=4.0/32,
  #                               height_shift_range=4.0/32,
  #                               fill_mode = "constant",
  #                               cval = 0,
  #                               horizontal_flip=True)
  val_gen = ImageDataGenerator()
  
  # train_gen = keras.Sequential(
  #   [
  #     layers.Resizing(size, size), 
  #     layers.RandomFlip("horizontal"),
      
  #   ]
  # )
  train_dataset = train_gen.flow(X_train, y_train, batch)
  val_dataset = val_gen.flow(X_test, y_test, batch)
  
  # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  # train_dataset = train_dataset.batch(batch).map(lambda x, y: (layers.Resizing(size,size)(x), y))
  
  # val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  # val_dataset = val_dataset.batch(batch).map(lambda x, y: (layers.Resizing(size,size)(x), y))
  
  return {'train': train_dataset, 'test': val_dataset}, {'train': 50000, 'test': 10000}, 32, 10