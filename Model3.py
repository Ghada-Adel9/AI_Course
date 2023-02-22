from re import X
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import itertools
import sndhdr
import time
from urllib import response
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# from PIL import Image
# import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras.layers import Conv1D,Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import warnings
from sklearn import metrics as ms
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
# for i in range(1, 21):
#     header += f' mfcc{i}'
# header += ' label'
# header = header.split()

# file = open('dataset.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
# Speakers = 'abdelrahman Baghdady ghada Hussen'.split()
# for g in Speakers:
#     for filename in os.listdir(f'dataset/train/{g}'):
#         songname = f'dataset/train/{g}/{filename}'
#         y, sr = librosa.load(songname, mono=True, duration=30)
#         rmse = librosa.feature.rms(y=y)[0]
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {g}'
#         file = open('dataset.csv', 'a', newline='')
#         with file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())

data = pd.read_csv('Ghada/dataset.csv')
data.head()
# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
#Encoding the Labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


##New
le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(le_name_mapping)
#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
##New
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
# # New
X_train = tf.keras.utils.normalize(X_train , axis=1)
X_test = tf.keras.utils.normalize(X_test , axis=1)

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(rate=0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
Our_Model = model.fit(X_train, y_train, batch_size=64, epochs=100,validation_data=(X_val, y_val),callbacks=[early_stop])


val_loss,val_accuracy = model.evaluate(X_test,y_test)
print(val_loss)
print(val_accuracy)

predictions = model.predict(X_test)
print("Prediction Shape is {}".format(predictions.shape))
print("X test = ", X_test[100] ,"Y test",y_test[100])

if y_test[100]== 0 :
    print("Baghdady")
elif y_test[100]== 1:
    print("Hussen")
elif y_test[100]== 3:
    print("ghada")
elif y_test[100]== 2:
    print("abdelrahman")
else :
    print("Unauthorized")

if np.argmax(predictions[100])== 0 :
    print("Baghdady")
elif np.argmax(predictions[100])== 1:
    print("Hussen")
elif np.argmax(predictions[100])== 2:
    print("abdelrahman")
elif np.argmax(predictions[100])== 3:
    print("ghada")
else :
    print("Unauthorized")

# file = open('data_Predict.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
# for filename in os.listdir(f'dataset/predict'):
#     songname = f'dataset/predict/{filename}'
#     y, sr = librosa.load(songname, mono=True, duration=30)
#     rmse = librosa.feature.rms(y=y)[0]
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
#     for e in mfcc:
#         to_append += f' {np.mean(e)}'
#     file = open('data_Predict.csv', 'a', newline='')
#     with file:
#         writer = csv.writer(file)
#         writer.writerow(to_append.split())


#test code anlyasis 


data_predict = pd.read_csv('Ghada/data_Predict.csv')
print(data_predict.head())
# Dropping unneccesary columns
data_predict = data_predict.drop(['filename'],axis=1)
print(data_predict.head())
XPredict = scaler.fit_transform(np.array(data_predict.iloc[:, :-1], dtype = float))
print(XPredict[0])
XPredict = tf.keras.utils.normalize(XPredict , axis=1)


start = time.perf_counter()
pred = model.predict(XPredict)
response_time = time.perf_counter()- start
print("Response Time = ",response_time)
print("X Predict = ", pred[0] )
if np.argmax( pred[0])== 0 :
    print("Baghdady")
elif np.argmax( pred[0])== 1:
    print("Hussen")
elif np.argmax(pred[0])== 2:
    print("abdelrahman")
elif np.argmax(pred[0])== 3:
    print("ghada")
else :
    print("Unauthorized")

fig, axs = plt.subplots(2)

    # create accuracy sublpot
axs[0].plot(Our_Model.history["accuracy"], label="train accuracy")
axs[0].plot(Our_Model.history["val_accuracy"], label="test accuracy")
axs[0].set_ylabel("Accuracy")
axs[0].legend(loc="lower right")
axs[0].set_title("Accuracy eval")

    # create error sublpot
axs[1].plot(Our_Model.history["loss"], label="train error")
axs[1].plot(Our_Model.history["val_loss"], label="test error")
axs[1].set_ylabel("Error")
axs[1].set_xlabel("Epoch")
axs[1].legend(loc="upper right")
axs[1].set_title("Error eval")

plt.show()

# To save model to use it later
model.save('VoiceRecognition.model')
# Extract model to use it again
new_model=tf.keras.models.load_model('VoiceRecognition.model')
# y_test=y_test.reshape((132))
# confusion_matrix(y_test,x_test)


 # get predictions on test set
test_predictions = np.argmax(model.predict(X_test),axis=-1)
# test_actual = np.argmax(y_test,axis=-1)
test_actual = y_test
print("\ntest_predictions::: ", len(test_predictions))
print("test_actual::: ", len(y_test))

# build confusion matrix and normalized confusion matrix
print('NN Confusion Matrix')
conf_matrix = confusion_matrix(test_actual, test_predictions)
conf_matrix_norm = confusion_matrix(test_actual, test_predictions,normalize='true')

# # set labels for matrix axes
unique_labels = np.unique(y_test)

 # make a confusion matrix with labels using a DataFrame
confmatrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=unique_labels, columns=unique_labels)


# plot confusion matrices   (16,6) // epochs
# MAKO
# top=0.95,
# bottom=0.06,
# left=0.08,
# right=1.0,
# hspace=0.23,
# wspace=0.2
plt.figure(figsize=(12,12))
sn.set(font_scale=1.2) # label and title size
plt.subplot(2, 1 ,1)
plt.title('NN | Confusion Matrix')
sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 9}, fmt='g') #annot_kws is value font
# fmt='g' --> turnoff the scientific notation
plt.subplot(2, 1 ,2)
plt.title('NN | Normalized Confusion Matrix')
sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 9}) #annot_kws is value font
plt.show()


# --------------------------------------------------------------------------------------------
# 
# TN, FP, FN, TP = ms.confusion_matrix(test_actual, test_predictions).ravel()
confM = ms.confusion_matrix (test_actual, test_predictions, labels=unique_labels)
print(confM)

FP = confM.sum(axis=0)-np.diag(confM)  
FN = confM.sum(axis=1)-np.diag(confM)
TP = np.diag(confM)
TN = confM.sum() - (FP + FN + TP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print("FP: ", FP)
print("FN: ", FN)
print("TP: ", TP)
print("TN: ", TN)
print("ACC: ", ACC)