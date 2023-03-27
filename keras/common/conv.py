from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.activations import relu

def Conv_BN_ReLU(ip, out_channels, kernel=(3,3), stride=(1,1), padding='same'):
    x = ip
    x = Conv2D(out_channels, kernel, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = relu(x)
    return x

def fc_ReLU_Drop(ip, out_channels):
    x = ip
    x = Dense(out_channels)(x)
    x = relu(x)
    x = Dropout(0.5)(x)
    return x