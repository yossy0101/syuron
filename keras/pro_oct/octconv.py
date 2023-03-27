from tensorflow.keras.layers import Conv2D, AveragePooling2D, UpSampling2D, MaxPooling2D, add, BatchNormalization
from tensorflow.keras.activations import relu

def OctConv2D(ip, out_channels, alpha=0.0, kernel=(3,3), stride=(1,1), padding='same'):
    ip_high, ip_low = ip
    
    out_lf = int(out_channels * alpha)
    out_hf = out_channels - out_lf
    
    x_h_h, x_h_l, x_l_h, x_l_l = None, None, None, None
    #HtoH
    if not (ip_high == None or out_hf == 0):
        x_h_h = Conv2D(out_hf, kernel, strides=stride, padding=padding)(ip_high)
    #HtoL
    if not (ip_high == None or out_lf == 0):
        x_h_l = AveragePooling2D()(ip_high)
        x_h_l = Conv2D(out_lf, kernel, strides=stride, padding=padding)(x_h_l)
    #LtoH    
    if not (ip_low == None or out_hf == 0):
        x_l_h = Conv2D(out_hf, kernel, strides=stride, padding=padding)(ip_low)
        x_l_h = UpSampling2D()(x_l_h)
    #LtoL
    if not (ip_low == None or out_lf == 0):
        x_l_l = Conv2D(out_lf, kernel, strides=stride, padding=padding)(ip_low)    
    
    if out_hf == 0:
        x_h = None
    elif x_h_h == None:
        x_h = x_l_h
    elif x_l_l == None:
        x_h = x_h_h
    else:
        x_h = x_h_h + x_l_h
        
    if out_lf == 0:
        x_l = None
    elif x_l_l == None:
        x_l = x_h_l
    elif x_h_l == None:
        x_l = x_l_l
    else:
        x_l = x_l_l + x_h_l
    
    return (x_h, x_l)

def Oct_BN_ReLU(ip, out_channels, alpha=0.0, kernel=(3,3), stride=(1,1), padding='same'):
    x_h, x_l = OctConv2D(ip, out_channels, alpha, kernel, stride, padding)
    if x_h is not None : x_h = BatchNormalization()(x_h)
    if x_h is not None : x_h = relu(x_h)
    if x_l is not None : x_l = BatchNormalization()(x_l)
    if x_l is not None : x_l = relu(x_l)
    return (x_h, x_l)

def Oct_MaxPool2D(ip, pool_size=(2,2), strides=(2,2)):
    x_h, x_l = ip
    
    if x_h is not None : x_h = MaxPooling2D(pool_size=pool_size, strides=strides)(x_h)
    if x_l is not None : x_l = MaxPooling2D(pool_size=pool_size, strides=strides)(x_l)
    return (x_h, x_l)