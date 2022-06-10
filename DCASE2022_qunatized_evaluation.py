import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pathlib
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from tensorflow.keras.models import load_model

from sklearn.metrics import log_loss
#%%
x_train=np.reshape(np.load('~/DCASE2022_numpy_mel_train_test_data/DCASE2022_train.npy'),[139620,40,51,1])
x_test=np.reshape(np.load('~/DCASE2022_numpy_mel_train_test_data/DCASE2022_test.npy'),[29680,40,51,1])
labels_test=np.load('~/DCASE2022_TASK1/DCASE2022_numpy_mel_train_test_data/label_test.npy')
labels_train=np.load('~/DCASE2022_numpy_mel_train_test_data/label_train.npy')


y_test = tf.keras.utils.to_categorical(labels_test, 10)
y_train=  tf.keras.utils.to_categorical(labels_train, 10)

#%%

os.chdir('~/Quantized_model/C123')   # set path of the quantized model.


prob_def = [ ]
interpreter = tf.lite.Interpreter(model_path="converted_quant_model_default.tflite")


all_tensor_details=interpreter.get_tensor_details()
interpreter.allocate_tensors()# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()# Test model on some input data.
input_shape = input_details[0]['shape']
# print(input_shape)



x_test=np.array(x_test,dtype=np.float32)
acc=0
for i in range(len(x_test)):
    input_data = x_test[i].reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob_def.append(output_data)
    if(np.argmax(output_data) == np.argmax(y_test[i])):
        acc+=1
acc_def = acc/len(x_test)
# print('default compressed acc:',acc_def*100)
logloss_def= log_loss(y_true=labels_test, y_pred= np.reshape(prob_def,[-1,10]), normalize=True)




print('..............................')
print('INT8: : ',acc_def*100,'logloss:  ', logloss_def)

print('..............................')
