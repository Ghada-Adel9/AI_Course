import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import  keras
import matplotlib.pyplot as plt
import time
import seaborn  as sn
from sklearn import metrics as ms


DATA_PATH = "data.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates RNN-LSTM model

    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    # model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))

    model.add(keras.layers.LSTM(256, input_shape=input_shape, return_sequences=True))

    # model.add(keras.layers.Dense(256, activation='relu', input_shape=input_shape))

    model.add(keras.layers.LSTM(256))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(12, activation='softmax'))

    return model



def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))



if __name__ == "__main__":

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



    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32,callbacks=[callback], epochs=50)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)


    # pick a sample to predict from the test set
    X_to_predict = X_test[150]
    y_to_predict = y_test[150]

    # predict sample
    predict(model, X_to_predict, y_to_predict)


# ------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

# def report(X_data, y_data): 
#     #Confution Matrix and Classification Report 
#     Y_pred = model.predict_classes(X_data) 
#     y_test_num = y_data.astype(np.int64) 
#     conf_mt = confusion_matrix(y_test_num, Y_pred) 
#     print(conf_mt) 
    
#     plt.matshow(conf_mt) 
#     plt.show() 
#     print('\nClassification Report') 
#     target_names = ["Jackson", "Nicola", "Theo", "Ankur", "Caroline", "Rodolfo", "Unknown"]
#     print(classification_report(y_test_num, Y_pred))


print('\n# TEST DATA #\n')
score = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# predict sample
start = time.perf_counter()
predict(model, X_to_predict, y_to_predict)
responseTime = time.perf_counter() - start

print(responseTime)

# get predictions on test set
test_predictions = np.argmax(model.predict(X_test),axis=-1)
# test_actual = np.argmax(y_test,axis=-1)
test_actual = y_test
print("\ntest_predictions::: ", len(test_predictions))
print("test_actual::: ", len(y_test))
# build confusion matrix and normalized confusion matrix
print('RNN Confusion Matrix')
conf_matrix = ms.confusion_matrix(test_actual, test_predictions)
conf_matrix_norm = ms.confusion_matrix(test_actual, test_predictions,normalize='true')
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
plt.title('RNN | Confusion Matrix')
sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 9}, fmt='g') #annot_kws is value font
# fmt='g' --> turnoff the scientific notation
plt.subplot(2, 1 ,2)
plt.title('RNN | Normalized Confusion Matrix')
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
