from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import clear_session
import os
import numpy as np
import time
from tqdm import *
import cv2
import TrainStep
import models

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:1], device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

img_path = '...\\Cholangiocarcinoma\\PreProcessed\\'
msk_path = '...\\Cholangiocarcinoma\\annotation\\'
model_results_dir = '...\\Cholangiocarcinoma\\models'
if not os.path.exists(model_results_dir):
    os.makedirs(model_results_dir)

img_resize = (256,192)
input_size = (img_resize[1],img_resize[0],60)
epochs = 100
batch_size = 8
learning_rate = 0.001

msk_files = [file for file in os.listdir(msk_path) if file.endswith(".png")]
msk_files.sort() 

X = []
Y = []

for j in trange(len(msk_files)):
    if j % 1 == 0:
        img_fl = msk_files[j]
        img_name = img_fl.split('.')[0]

        image = cv2.imread(img_path + '\\{}\\0.png'.format(img_name), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, img_resize)

        # cv2.imshow('img',image)
        # cv2.waitKey(0)

        for i in range(1,60):
            hdr_Slice = cv2.imread(img_path + '\\{}\\{}.png'.format(img_name,i), cv2.IMREAD_GRAYSCALE)
            hdr_Slice = cv2.resize(hdr_Slice, img_resize)
            # cv2.imshow('img',hdr_Slice)
            # cv2.waitKey(0)
            image = np.dstack((image, hdr_Slice))
        
        X.append(image)

        mask = cv2.imread(msk_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_resize)
        # mask = cv2.flip(mask,0)
        # cv2.imshow('img',mask)
        # cv2.waitKey(0)

        Y.append(mask)
        
Y = Y.reshape((Y.shape[0],Y.shape[1],Y.shape[2],1))

np.random.seed(1000)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
y_shuffled = Y[shuffle_indices]

x_shuffled = x_shuffled / 255
y_shuffled = y_shuffled / 255
y_shuffled = np.round(y_shuffled,0)

print(x_shuffled.shape)
print(y_shuffled.shape)

length = int(float(len(x_shuffled))/5)

for i in range(0,1):
    tic = time.ctime()
    fp = open(model_results_dir +'\\jaccard-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\dice-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()

    index = int(float(len(x_shuffled))*(i+1)/5)
    x_train = np.concatenate((x_shuffled[:index-length], x_shuffled[index:]), axis=0)
    x_val = x_shuffled[index-length:index]
    y_train = np.concatenate((y_shuffled[:index-length], y_shuffled[index:]), axis=0)
    y_val = y_shuffled[index-length:index]

    model = models.SAHISNet(pretrained_weights = None,input_size = input_size)
    model.summary()

    print ('iter: %s' % (str(i)))

    TrainStep.trainStep(model, X_train=x_train, Y_train=y_train, X_test=x_val, Y_test=y_val, epochs=epochs, batch_size=batch_size, iters=i, results_save_path=model_results_dir)

    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-jaccard.txt','a')
    tic = time.ctime()
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-dice.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()
    
    clear_session()
    tf.compat.v1.reset_default_graph()
