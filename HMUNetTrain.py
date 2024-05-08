from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
import models
import os
import numpy as np
import time
from tqdm import *
import cv2

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:1], device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Split the original hyperspectral image into 60 images, each image is histogram matched according to the corresponding 27th band, and stored in the img_path_1 folder according to the file name
img_path_1 = '...\\Cholangiocarcinoma\\SplitHistogramMatched'

# Split the original hyperspectral image into 60 images and store them in the img_path_1 folder according to the file name
img_path_2 = '...\\Cholangiocarcinoma\\Split'

img_resize = (256,192)
input_size = (192,256,1)

X1 = []
X2 = []
Y = []

sub_dirs = next(os.walk(img_path_2))[1]
for dir in tqdm(sub_dirs):
    if dir.split('-')[-1] != 'blank':
        if os.path.exists(img_path_1 + '\\' + dir + '\\27.png'):
            list1 = [1,5,10,15,20,25,30,35,40,45,50,55]
            for i in list1:
                img = cv2.imread(img_path_2 + '\\' + dir + '\\{}.png'.format(i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_resize)
                img = img / 255
                X1.append(img)

                img = cv2.imread(img_path_2 + '\\' + dir + '\\27.png', cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_resize)
                img = img / 255
                X2.append(img)

                img = cv2.imread(img_path_1 + '\\' + dir + '\\{}.png'.format(i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_resize)
                img = img / 255
                Y.append(img)

X1 = np.array(X1)
X2 = np.array(X2)
Y = np.array(Y)

X1 = X1.reshape((X1.shape[0], X1.shape[1], X1.shape[2],1))  # Grayscale input images
X2 = X2.reshape((X2.shape[0], X2.shape[1], X2.shape[2],1))
Y = Y.reshape((Y.shape[0], Y.shape[1], Y.shape[2],1))

np.random.seed(1000)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x1_shuffled = X1[shuffle_indices]
x2_shuffled = X2[shuffle_indices]
y_shuffled = Y[shuffle_indices]

length = int(float(len(x1_shuffled))/5)
index = int(float(len(x1_shuffled))*(0+1)/5)
x1_train = np.concatenate((x1_shuffled[:index-length], x1_shuffled[index:]), axis=0)
x1_val = x1_shuffled[index-length:index]
x2_train = np.concatenate((x2_shuffled[:index-length], x2_shuffled[index:]), axis=0)
x2_val = x2_shuffled[index-length:index]
y_train = np.concatenate((y_shuffled[:index-length], y_shuffled[index:]), axis=0)
y_val = y_shuffled[index-length:index]

model = models.HMUNet(pretrained_weights=None, input_size=input_size)
model.summary()

callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='mse', 
        patience=20
    ),
    tf.keras.callbacks.ModelCheckpoint(
    filepath = '...\\Cholangiocarcinoma\\models\\HMUNet.h5', 
    monitor='val_loss', 
    save_best_only=True) 
]

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mse',ssim_metric])

model.fit(x=[x1_train, x2_train], y=y_train, validation_data=([x1_val, x2_val], y_val), batch_size=8,callbacks=callbacks_list, epochs=100, verbose=1)