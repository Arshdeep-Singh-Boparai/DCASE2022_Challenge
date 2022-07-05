#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:24:19 2022

@author: arshdeep
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint

#%%
# Configuration options
# feature_vector_length = 51
num_classes = 10


import os
os.chdir('/home/arshdeep/Pruning/DCASE2022_TASK1/audio_data/DCASE2022_TASK1/DCASE2022_numpy_mel_train_test_data')
x_train=np.reshape(np.load('DCASE2022_train.npy'),[139620,40,51,1])
x_test=np.reshape(np.load('DCASE2022_test.npy'),[29680,40,51,1])
labels_test=np.load('labels_test.npy')
labels_train=np.load('label_train.npy')


# X_train = np.reshape(np.average(x_train,2),[139620,40])
# X_test  = np.reshape(np.average(x_test,2),[29680,40])



# Load the data

# Reshape the data - MLPs do not understand such things as '2D'.
# Reshape to 28 x 28 pixels = 784 features
# X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
# X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

# # Convert into greyscale
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# Convert target classes to categorical ones
Y_train = to_categorical(labels_train, num_classes)
Y_test = to_categorical(labels_test, num_classes)

size_batch =60


#%% time model
# Set the input shape
feature_vector_length = 40

X_train= np.reshape(np.average(x_train,2),[139620,40])
X_test= np.reshape(np.average(x_test,2),[29680,40])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



input_shape = (feature_vector_length,)
print(f'Feature shape: {input_shape}')

# Create the model
model = Sequential()
model.add(Dense(100, input_shape=input_shape, activation='tanh'))
# layer_BN3 =  BatchNormalization() #layer2
# model.add(layer_BN3)
model.add(Dense(50, activation='tanh'))
# model.add(Dense(50, activation='tanh'))
model.add(Dense(num_classes, activation='softmax'))

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

os.chdir('/home/arshdeep/Pruning/DCASE2022_TASK1/MLP_classification/')

checkpointer = ModelCheckpoint(filepath='best_weights_dcase2021_L2.h5py',monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)
model.load_weights('best_weights_dcase2021_L2.h5py')
hist=model.fit(X_train,Y_train,batch_size=64,epochs=300,verbose=1,validation_data=(X_test, Y_test),callbacks=[checkpointer])
# 
model.load_weights('best_weights_dcase2021_L2.h5py')

model.save('best_model_dcase2021_L2.h5')
# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
# model.fit(epochs=EPOCHS, )

# Test the model after training
test_results = model.evaluate(X_test, Y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
from sklearn.metrics import log_loss
logloss_overall = log_loss(y_true=labels_test, y_pred=model.predict(X_test), normalize=True)

print(logloss_overall)
#%% frequency model

# feature_vector_length = 51

# X_train_f = np.reshape(np.average(x_train,1),[139620,51])
# X_test_f = np.reshape(np.average(x_test,1),[29680,51])

# X_train_f = X_train_f.astype('float32')
# X_test_f = X_test_f.astype('float32')
# X_train /= 255
# X_test /= 255

# input_shape = (feature_vector_length,)
# print(f'Feature shape: {input_shape}')

# # Create the model
# model_f = Sequential()
# model_f.add(Dense(100, input_shape=input_shape, activation='tanh'))
# model_f.add(Dense(50, activation='relu'))
# model_f.add(Dense(num_classes, activation='softmax'))

# # Configure the model and start training
# model_f.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_f.fit(X_train_f, Y_train, epochs=40, batch_size=size_batch, verbose=1, validation_split=0.2)

# # Test the model after training
# test_results_f = model_f.evaluate(X_test_f, Y_test, verbose=1)
# print(f'Test results - Loss: {test_results_f[0]} - Accuracy: {test_results_f[1]}%')


#%%

# prob_t = model.predict_proba(X_test)
# prob_f = model_f.predict_proba(X_test_f)


# prob_add = np.array(prob_t)  + np.array(prob_f)


# predd_class = np.argmax(prob_add,1)


# asd=confusion_matrix(labels_test,predd_class);
# accu=(np.trace(asd)/len(labels_test))*100;
# print(accu)








