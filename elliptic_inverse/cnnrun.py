#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:28:15 2020

@author: apple
"""

import tensorflow as tf

import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
import os


def datacnn(x_train,y_train,d,eval=False,x_test=None,y_test=None):
    x_train = (x_train).reshape((-1, int(np.sqrt(d)), int(np.sqrt(d)), 1)).astype('float32')
    if x_test is not None:
        x_test = (x_test).reshape((-1, int(np.sqrt(d)), int(np.sqrt(d)), 1)).astype('float32')
    #y_train = (y_train).reshape(-1,m)
    weight_init = RandomNormal()
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init, input_shape=(np.sqrt(d), np.sqrt(d), 1)))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=weight_init))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=weight_init))
    #model.add(tf.keras.activations.relu(x, alpha=0.01, max_value=None))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='softmax', kernel_initializer=weight_init))
    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # fit the model
    model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=2)
    loss_object = tf.keras.losses.MeanSquaredError()
    # evaluate the model
    if eval==True:
        #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33)
        loss = model.evaluate(x_test, y_test, verbose=2)
        print('Loss for test: %.3f' % loss)
    x_train=tf.Variable(x_train,trainable=True)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_train)
        pred_y = model(x_train)
        # Calculate loss
        model_loss = loss_object(y_train, pred_y)
        
    # Calculate gradients
    model_gradients = tape.gradient(pred_y, x_train)
    model_gradientsloss = tape.gradient(model_loss, x_train)
    return model_gradients,model_gradientsloss,model




if __name__ == "__main__": 

    algs=['EKI','EKS']
    ensbl_sz = 100
    path = '/Users/apple/Downloads/'
    loaded=np.load(file=os.path.join(path,algs[0]+'_ensbl'+str(ensbl_sz)+'_training.npz'))
    X=loaded['X']
    Y=loaded['Y']
    ratio = 0.75
    n = X.shape[0]
    d = 41**2
    X = tf.keras.utils.normalize(X, axis=1, order=2)  #keep each image in same scale
    x_train,x_test,y_train,y_test = X[:int(n*ratio)],X[int(n*ratio):],Y[:int(n*ratio)],Y[int(n*ratio):]
    model_gradients,model_gradientsloss,model = datacnn(x_train,y_train,d,True,x_test,y_test)