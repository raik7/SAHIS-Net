import evaluate
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
import tensorflow as tf
from tqdm import *

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batch_size, iters, results_save_path):
    optimizer2 = Adam(learning_rate=0.001)

    first_layer_group = []
    second_layer_group = []

    for layer in model.layers:
        print(layer.name)
        if layer.name.startswith('sa'):
            for var in layer.trainable_variables:
                first_layer_group.append(var)
        else:
            for var in layer.trainable_variables:
                second_layer_group.append(var)

    nBatch = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        for i in range(nBatch):
            with tf.GradientTape(persistent=True) as tape:
                xBatch = X_train[batch_size*i:batch_size*(i+1),:,:,:]
                yBatch = Y_train[batch_size*i:batch_size*(i+1),:,:,:]
                predictions = model(xBatch, training=True)
                loss = 0.4 * tf.losses.binary_crossentropy(yBatch, predictions[0]) + 0.6 * tf.losses.binary_crossentropy(yBatch, predictions[1])
                # print(np.mean(loss))
            
            gradients_first_group = tape.gradient(loss, first_layer_group)
            gradients_second_group = tape.gradient(loss, second_layer_group)

            lr = cosine_decay_with_warmup(global_step = epoch,
                             learning_rate_base = 0.1,
                             total_steps = epochs,
                             warmup_learning_rate = 0,
                             warmup_steps=0,
                             hold_base_rate_steps=0)

            optimizer1 = Adam(learning_rate=lr)
            

            optimizer1.apply_gradients(zip(gradients_first_group, first_layer_group))
            optimizer2.apply_gradients(zip(gradients_second_group, second_layer_group))

        evaluate.evaluateModel(model, X_test, Y_test, batch_size, iters, results_save_path)