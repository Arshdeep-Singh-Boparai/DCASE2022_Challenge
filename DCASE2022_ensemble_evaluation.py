import numpy as np    
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix


import os



from sklearn.metrics import log_loss

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


x_train=np.reshape(np.load('~/DCASE2022_numpy_mel_train_test_data/DCASE2022_train.npy'),[139620,40,51,1])
x_test=np.reshape(np.load('~/DCASE2022_numpy_mel_train_test_data/DCASE2022_test.npy'),[29680,40,51,1])
labels_test=np.load('~/DCASE2022_numpy_mel_train_test_data/label_test.npy')
labels_train=np.load('~/DCASE2022_numpy_mel_train_test_data/label_train.npy')


y_test = tf.keras.utils.to_categorical(labels_test, 10)
y_train=  tf.keras.utils.to_categorical(labels_train, 10)

list_dir = np.load('/~/DCASE2022_numpy_mel_train_test_data/feature_list_test.npy')

#%% load probabilites (INT 8)


unpruned_int8  = np.load('~/Quantized_model/unpruned/prob_def.npy')
Pruned_C1_int8 = np.load('~/Quantized_model/C1/prob_def.npy')  
Pruned_C2_int8 = np.load('~/Quantized_model/C2/prob_def.npy')  
Pruned_C3_int8 = np.load('~/Quantized_model/C3/prob_def.npy')  
Pruned_C12_int8 = np.load('~/Quantized_model/C12/prob_def.npy')  
Pruned_C23_int8 = np.load('~/Quantized_model/C23/prob_def.npy')  
Pruned_C123_int8 = np.load('~/Quantized_model/C123/prob_def.npy')  


# unpruned_float16  = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/unpruned/prob_def.npy')
# Pruned_C1_float16 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C1/prob_16.npy')  
# Pruned_C2_float16 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C2/prob_16.npy')  
# Pruned_C3_float16= np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C3/prob_16.npy')  
# Pruned_C12_float16 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C12/prob_16.npy')  
# Pruned_C23_float16= np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C23/prob_16.npy')  
# Pruned_C123_float16 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C123/prob_16.npy')  



# unpruned_float32  = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/unpruned/prob_def.npy')
# Pruned_C1_float32 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C1/prob_32.npy')  
# Pruned_C2_float32= np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C2/prob_32.npy')  
# Pruned_C3_float32= np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C3/prob_32.npy')  
# Pruned_C12_float32 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C12/prob_32.npy')  
# Pruned_C23_float32= np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C23/prob_32.npy')  
# Pruned_C123_float32 = np.load('/home/arshdeep/Pruning/DCASE2022_TASK1/DCASE2022_challenge_pruned_networks/cosine_similarity/quantized_model/C123/prob_32.npy')  







#%%


# pred_label_float32 = (Pruned_C1_float32 + Pruned_C2_float32 + Pruned_C3_float32 + Pruned_C12_float32 + Pruned_C23_float32 + Pruned_C123_float32)/6
# pred_label_float16= (Pruned_C1_float16 + Pruned_C2_float16 + Pruned_C3_float16 + Pruned_C12_float16 + Pruned_C23_float16 + Pruned_C123_float16)/6
pred_label_int8= (Pruned_C1_int8 + Pruned_C2_int8 + Pruned_C3_int8 + Pruned_C12_int8 + Pruned_C23_int8 + Pruned_C123_int8)/6

# pred_label_int8= (Pruned_C1_int8 + Pruned_C2_int8 + Pruned_C3_int8 + Pruned_C12_int8 + Pruned_C23_int8)/5


pred_label = pred_label_int8

pred=np.argmax(pred_label,1)
asd=confusion_matrix(labels_test,pred,normalize='true');
accu=(np.trace(asd))*10#/np.size(labels_test))*100;

logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label, normalize=True)

print(logloss_overall, 'baseline--log loss')
print(accu,'baseline accuracy')

target_names = ['airport','bus','shopping_mall','metro','street_pedestrian','public_square','street_traffic','tram','metro_station','park']
print(classification_report(labels_test,pred,target_names=target_names))

 



plt.figure(figsize=(15,9))
array1=asd#[[309,0,0,0,0,0,26,2,0,0,84],[0,390,0,0,0,0,0,0,0,16,9],[0,2,403,3,0,0,1,0,0,7,17],[2,0,13,370,0,3,4,0,0,2,41],[0,0,0,0,358,5,0,0,0,0,23],[0,0,0,2,0,307,10,0,3,0,65],[48,0,0,0,0,0,286,1,0,0,106],[25,0,0,0,0,40,2,82,0,0,280],[0,0,0,0,0,0,2,0,395,0,5],[0,11,5,1,0,1,0,0,0,365,53],[0,0,157,31,0,1,0,0,0,53,103]]
df_cm = pd.DataFrame(array1, target_names,
                  target_names)
#plt.figure(figsize = (10,7))
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size":17},fmt='.0%',cmap="YlGnBu")# font size
plt.xlabel('Predicted')
plt.ylabel('Actual')
