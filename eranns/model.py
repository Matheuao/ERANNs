import tensorflow.keras as keras
from .layers import ARB

def get_stride_size(sm, i):
        if i > sm:
            return 2
            
        elif i <= sm:
            return 4
            
        else:
            print("i must be between 0 and 3")
            return None

def ERANNs(sm, W, T0 = 128, N = 527):
    s1 = get_stride_size(sm, 1)
    s2 = get_stride_size(sm, 2)
    s3 = get_stride_size(sm, 3)
    
    model = keras.Sequential([
        # feature extratorc, batch normalization
        keras.layers.Input(shape=(128, T0, 1)),
        # stage 0
        ARB(stride_freq=1, stride_time=1, channels=8*W),
        ARB(stride_freq=1, stride_time=1, channels=8*W),
        ARB(stride_freq=1, stride_time=1, channels=8*W),
        ARB(stride_freq=1, stride_time=1, channels=8*W),
        #stage 1
        ARB(stride_freq=2, stride_time=s1, channels=16*W), 
        ARB(stride_freq=1, stride_time=1, channels=16*W),
        ARB(stride_freq=1, stride_time=1, channels=16*W),
        ARB(stride_freq=1, stride_time=1, channels=16*W),
        #stage 2
        ARB(stride_freq=2, stride_time=s2, channels=32*W),
        ARB(stride_freq=1, stride_time=1, channels=32*W),
        ARB(stride_freq=1, stride_time=1, channels=32*W),
        ARB(stride_freq=1, stride_time=1, channels=32*W),
        #stage 3
        ARB(stride_freq=2, stride_time=s3, channels=64*W),
        ARB(stride_freq=1, stride_time=1, channels=64*W),
        ARB(stride_freq=1, stride_time=1, channels=64*W),
        ARB(stride_freq=1, stride_time=1, channels=64*W),
        #stage 4
        ARB(stride_freq=2, stride_time=2, channels=128*W),
        ARB(stride_freq=1, stride_time=1, channels=128*W),
        ARB(stride_freq=1, stride_time=1, channels=128*W),
        ARB(stride_freq=1, stride_time=1, channels=128*W),
        #global pooling
        keras.layers.GlobalAveragePooling2D(), # mudar depois         
        #stage 5
        keras.layers.Dense(128*W),#FC1
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Dense(N, activation = "softmax"),#FC2
    ])
    

