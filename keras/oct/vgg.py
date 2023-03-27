import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Resizing
from tensorflow.keras.activations import softmax
from conv import Conv_BN_ReLU, fc_ReLU_Drop
from octconv import Oct_BN_ReLU, Oct_MaxPool2D

def _make_layer(ip, out_channels, blocks, alpha=0.25, output=False):
  x = ip
  for i in range (blocks):
    alpha = 0.0 if output==True else alpha
    x = Oct_BN_ReLU(x, out_channels, alpha)
  x = Oct_MaxPool2D(x, pool_size=(2,2), strides=(2,2))
  return x
  
def VGG(layers, defo_image_size, image_size, num_classes=100):
  ip = Input(shape=(image_size,image_size,3))
  x = _make_layer((ip, None), 64, layers[0])
  x = _make_layer(x, 128, layers[1])
  x = _make_layer(x, 256, layers[2])
  x = _make_layer(x, 512, layers[3])
  x_h, x_l = _make_layer(x, 512, layers[4], output=True)
  x = GlobalAveragePooling2D()(x_h)
  x = Dropout(0.5)(x)
  x = fc_ReLU_Drop(x, 4096)
  x = fc_ReLU_Drop(x, 4096)
  x = Dense(num_classes)(x)
  x = softmax(x)
  return Model(ip, x)
  
def _vgg(layers, defo_image_size, image_size, num_classes, **kwargs):
  model = VGG(layers, defo_image_size, image_size, num_classes, **kwargs)
  return model