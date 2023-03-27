import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Resizing
from tensorflow.keras.activations import softmax
from conv import Conv_BN_ReLU, fc_ReLU_Drop

def _make_layer(ip, out_channels, blocks, kernel=(3,3), stride=(1,1), padding='same', input=False):
  x = ip
  for i in range (blocks):
    x = Conv_BN_ReLU(x, out_channels)
  x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
  return x
  
def VGG(layers, defo_image_size, image_size, num_classes=100):
  ip = Input(shape=(image_size,image_size,3))
  x = _make_layer(ip, 64, layers[0], input=True)
  x = _make_layer(x, 128, layers[1], input=True)
  x = _make_layer(x, 256, layers[2], input=True)
  x = _make_layer(x, 512, layers[3], input=True)
  x = _make_layer(x, 512, layers[4], input=True)
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  x = fc_ReLU_Drop(x, 4096)
  x = fc_ReLU_Drop(x, 4096)
  x = Dense(num_classes)(x)
  x = softmax(x)
  return Model(ip, x)
  
def _vgg(layers, defo_image_size, image_size, num_classes, **kwargs):
  model = VGG(layers, defo_image_size, image_size, num_classes, **kwargs)
  return model