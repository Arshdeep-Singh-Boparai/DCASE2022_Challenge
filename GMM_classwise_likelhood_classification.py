#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:44:12 2022

@author: arshdeep
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import os
from sklearn.metrics import classification_report,confusion_matrix
import pickle
from sklearn.metrics import log_loss

#%% data load
os.chdir('/home/arshdeep/Pruning/DCASE2022_TASK1/audio_data/DCASE2022_TASK1/DCASE2022_numpy_mel_train_test_data')
X_train=np.reshape(np.load('DCASE2022_train.npy'),[139620,40,51,1])
X_test=np.reshape(np.load('DCASE2022_test.npy'),[29680,40,51,1])
y_test=np.load('labels_test.npy')
y_train=np.load('label_train.npy')


# train_data = np.reshape(X_train, [139620,40*51]) # training data
# train_label = y_train                           # training labels

# test_data = np.reshape(X_test, [29680,40*51])
# test_label =y_test

train_data = np.reshape(np.average(X_train,2), [139620,40]) # training data
train_label = y_train                           # training labels

test_data = np.reshape(np.average(X_test,2), [29680,40])
test_label =y_test

n_classes =10
n_comp=50
#%% train GMM classwise
os.chdir('/home/arshdeep/Pruning/DCASE2022_TASK1/GMM_classification/GMM_classwise/')
GMM_dict = {"Class":[],"GMM_model":[]};

for i in range(n_classes):
    X  = train_data[train_label == i]
    gm = GaussianMixture(n_components=n_comp, random_state=0).fit(X)
    GMM_dict['Class'].append(i)
    GMM_dict['GMM_model'].append(gm)
    gm= [ ]
    print(i, ' GMM trained')


#%%




# a_file = open("GMM_dict_15.pkl", "wb")

# pickle.dump(GMM_dict, a_file)

# a_file.close()
# #%%



class_score = []  # GMM score for a given class
sample_score = []  # all GMM scores
pred_class = [ ]

for i in range(len(test_label)):
    test_sample = test_data[i].reshape([1,-1])
    for j in range(n_classes):
        pred= GMM_dict['GMM_model'][j].score_samples(test_sample)
        class_score.append(pred)
    sample_score.append(class_score) 
    pred_class.append(np.argmax(class_score))
    class_score = [ ]
    print(i, ' sample prediction done')
        
        
        
asd=confusion_matrix(test_label,pred_class);
accu=(np.trace(asd)/len(test_label))*100;
print(accu)

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# gm = GaussianMixture(n_components=3, random_state=0).fit(X)
# gm.means_


# gm.predict([[0, 0], [12, 3]])

#%%
# pred_prob_1 = np.reshape(sample_score,[29680,10])
# logloss_overall = log_loss(y_true=y_test, y_pred=pred_prob_1, normalize=True)