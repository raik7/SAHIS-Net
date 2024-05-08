import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras import backend as K


class Mish(Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def conv2d_norm(x, filters, kernel_size=(3, 3), padding='same', groups=1, strides=(1, 1), activation=None, regularizer = None, norm = 'bn',name=None):

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups,use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'zeros', kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    if norm == 'bn':
        x = BatchNormalization()(x)
    elif norm == 'ln':
        x = LayerNormalization()(x)
    # x = BatchNormalization(axis = 3, scale = True)(x)

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

def CFE(x):
    reshape_size = (1,1,x.shape[-1])
    filters = x.shape[-1]

    cw = GlobalAveragePooling2D()(x)
    cw = Dense(filters, activation='relu', kernel_initializer = initializers.RandomNormal(stddev=0.0000001), bias_initializer = 'ones')(cw)
    cw = Dense(filters, activation='sigmoid', kernel_initializer = initializers.RandomNormal(stddev=0.0000001), bias_initializer = 'ones')(cw)
    cw = Reshape(reshape_size)(cw)
    out = multiply([x,cw])

    return out

def SFE(x):
    filters = x.shape[-1]

    sw = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(x)
    sw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(sw)
    out = multiply([x,sw])

    return out

def FE(x):
    Sr = x

    Sa = CFE(x)
    Sa = SFE(Sa)

    So = add([Sa, Sr])

    return So

def HMUNet(pretrained_weights = None,input_size = (256,256,1)):
    kn=24

    input1 = Input(input_size)
    input2 = Input(input_size)
    # conv1 = conv2d_norm(inputs, kn, 7,'same',activation='relu')

    conv1_1 = conv2d_norm(input1, kn, 3,'same',activation='relu')
    conv1_1 = conv2d_norm(conv1_1, kn, 3,'same',activation='relu')
 
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    conv2_1 = conv2d_norm(pool1_1, kn*2, activation='relu')
    conv2_1 = conv2d_norm(conv2_1, kn*2, activation='relu')
 
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
    conv3_1 = conv2d_norm(pool2_1, kn*4, activation='relu')
    conv3_1 = conv2d_norm(conv3_1, kn*4, activation='relu')
 
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
    conv4_1 = conv2d_norm(pool3_1, kn*8, activation='relu')
    conv4_1 = conv2d_norm(conv4_1, kn*8, activation='relu')
 
    conv1_2 = conv2d_norm(input2, kn, 3,'same',activation='relu')
    conv1_2 = conv2d_norm(conv1_2, kn, 3,'same',activation='relu')
 
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2_2 = conv2d_norm(pool1_2, kn*2, activation='relu')
    conv2_2 = conv2d_norm(conv2_2, kn*2, activation='relu')
 
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3_2 = conv2d_norm(pool2_2, kn*4, activation='relu')
    conv3_2 = conv2d_norm(conv3_2, kn*4, activation='relu')
 
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    conv4_2 = conv2d_norm(pool3_2, kn*8, activation='relu')
    conv4_2 = conv2d_norm(conv4_2, kn*8, activation='relu')

    conv1 = concatenate([conv1_1,conv1_2], axis = 3)
    conv2 = concatenate([conv2_1,conv2_2], axis = 3)
    conv3 = concatenate([conv3_1,conv3_2], axis = 3)
    conv4 = concatenate([conv4_1,conv4_2], axis = 3)

    up7 = Conv2D(kn*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = conv2d_norm(merge7, kn*4, activation='relu')
    conv7 = conv2d_norm(conv7, kn*4, activation='relu')

    up8 = Conv2D(kn*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = conv2d_norm(merge8, kn*2, activation='relu')
    conv8 = conv2d_norm(conv8, kn*2, activation='relu')

    up9 = Conv2D(kn, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = conv2d_norm(merge9, kn, activation='relu')
    conv9 = conv2d_norm(conv9, kn, activation='relu')
    # conv9 = conv2d_norm(conv9, 2, activation='relu')
    conv10 = conv2d_norm(conv9, 1, activation='relu')

    model = Model(inputs = [input1, input2], outputs = conv10)
    # model.summary()
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def SAHISNet(pretrained_weights = None,input_size = (256,256,1)):
    kn=16

    inputs = Input(input_size)

    sa = DepthwiseConv2D(kernel_size = (1,1), depthwise_initializer= 'ones', use_bias=False, name = 'sa')(inputs)

    conv1 = conv2d_norm(sa, kn, 3,'same',activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1_FE = FE(conv1)

    conv2 = conv2d_norm(pool1, kn*2, 3,'same',activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2_FE = FE(conv2)

    conv3 = conv2d_norm(pool2, kn*4, 3,'same',activation='relu')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3_FE = FE(conv3)

    conv4 = conv2d_norm(pool3, kn*8, 3,'same',activation='relu')
    conv4_FE = FE(conv4)

    up5 = Conv2D(kn*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4_FE))
    merge5 = concatenate([conv3_FE,up5], axis = 3)
    conv5 = conv2d_norm(merge5, kn*4, activation='relu')
    conv5_FE = FE(conv5)

    up6 = Conv2D(kn*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2_FE,up6], axis = 3)
    conv6 = conv2d_norm(merge6, kn*2, activation='relu')
    conv6_FE = FE(conv6)

    up7 = Conv2D(kn, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1_FE,up7], axis = 3)
    conv7 = conv2d_norm(merge7, kn, activation='relu')

    out1 = Conv2D(1, (1, 1), activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal', name = 'out1')(conv7)

    input2 = concatenate([sa,out1], axis = 3)

    conv8 = conv2d_norm(input2, kn, 3,'same',activation='relu')
    pool8 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv8_FE = FE(conv8)
    merge8 = concatenate([pool8, conv6_FE], axis = 3)

    conv9 = conv2d_norm(merge8, kn*2, 3,'same',activation='relu')
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)
    conv9_FE = FE(conv9)
    merge9 = concatenate([pool9, conv5_FE], axis = 3)

    conv10 = conv2d_norm(merge9, kn*4, 3,'same',activation='relu')
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
    conv10_FE = FE(conv10)
    merge10 = concatenate([pool10, conv4_FE], axis = 3)

    conv11 = conv2d_norm(merge10, kn*8, 3,'same',activation='relu')
    conv11_FE = FE(conv11)

    up12 = Conv2D(kn*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv11_FE))
    merge12 = concatenate([conv10_FE,up12], axis = 3)
    conv12 = conv2d_norm(merge12, kn*4, activation='relu')

    up13 = Conv2D(kn*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv12))
    merge13 = concatenate([conv9_FE,up13], axis = 3)
    conv13 = conv2d_norm(merge13, kn*2, activation='relu')

    up14 = Conv2D(kn, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv13))
    merge14 = concatenate([conv8_FE,up14], axis = 3)
    conv14 = conv2d_norm(merge14, kn, activation='relu')

    out2 = Conv2D(1, (1, 1), activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal', name = 'out2')(conv14)

    model = Model(inputs = inputs, outputs = [out1, out2])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



# model = SAHISNet(pretrained_weights = None,input_size = (192,256,60))
# for layer in model.layers:
#         print(layer.name)
# model.summary()
