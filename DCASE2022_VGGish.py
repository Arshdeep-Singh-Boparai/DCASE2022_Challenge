# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:07:53 2022

@author: Arshdeep Singh
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:35:36 2021

@author: arshdeep
"""


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint




#%% data loading

num_classes = 10


import os
os.chdir('/home/arshdeep/Pruning/DCASE2022_TASK1/audio_data/DCASE2022_TASK1/DCASE2022_numpy_mel_train_test_data')
X_train=np.reshape(np.load('DCASE2022_train.npy'),[139620,40,51,1])
X_test=np.reshape(np.load('DCASE2022_test.npy'),[29680,40,51,1])
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

# X_train= np.reshape(np.average(x_train,2),[139620,40])
# X_test= np.reshape(np.average(x_test,2),[29680,40])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


#%%model building

input_shape = (40, 51, 1)
aud_input = Input(shape=input_shape, name='input_1')

# Block 1
x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

# Block 2
x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

# Block 3
x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

# Block 4
x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)


model = Model(aud_input, x, name='VGGish_dens1')

Z = list(np.load('/home/arshdeep/Pruning/VGGish/VGGish_dense1_weights.npy',allow_pickle=True)[0:4])
model.set_weights(Z)

x = GlobalAveragePooling2D(name='Global_Avg_Pooling')(x)

d1 = Dense(128, activation='relu', name='vggish_dense2')(x)
out = Dense(10, activation='softmax', name='vggish_out')(d1)


model_all = Model(aud_input, out, name='VGGish_dens1')


#%%

model_all.compile(loss='MeanAbsoluteError', optimizer='adam', metrics=['accuracy'])

os.chdir('/home/arshdeep/Pruning/DCASE2022_TASK1/VGGish_classification/')

checkpointer = ModelCheckpoint(filepath='best_weights_dcase2021_L2.h5py',monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)

hist=model_all.fit(X_train,Y_train,batch_size=32,epochs=1,verbose=1,validation_data=(X_test, Y_test),callbacks=[checkpointer])
# 
model_all.load_weights('best_weights_dcase2021_L2.h5py')

model_all.save('best_model_dcase2021_L2.h5')
# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
# model.fit(epochs=EPOCHS, )

# Test the model after training
test_results = model_all.evaluate(X_test, Y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
from sklearn.metrics import log_loss
logloss_overall = log_loss(y_true=labels_test, y_pred=model.predict(X_test), normalize=True)

print(logloss_overall)

