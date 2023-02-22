import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import time
import pandas as pd
import seaborn as sn
from sklearn import metrics as ms

# # path to json file that stores MFCCs and genre labels for each processed segment
# DATA_PATH = "data.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

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

    # # load data
    # X, y = load_data(DATA_PATH)

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

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = Sequential([

        # input layer
        keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(12, activation='softmax')
    ])

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50
    )

    # plot accuracy and error as a function of the epochs
    plot_history(history)


    # pick a sample to predict from the test set
    X_to_predict = X_test[3]
    y_to_predict = y_test[3]

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
print('Confusion Matrix')
conf_matrix = ms.confusion_matrix(test_actual, test_predictions)
conf_matrix_norm = ms.confusion_matrix(test_actual, test_predictions,normalize='true')
# # set labels for matrix axes
unique_labels = np.unique(y_test)
# make a confusion matrix with labels using a DataFrame
confmatrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=unique_labels, columns=unique_labels)

# plot confusion matrices   (16,6) // epochs
# top=0.95,
# bottom=0.06,
# left=0.08,
# right=1.0,
# hspace=0.23,
# wspace=0.2
plt.figure(figsize=(12,12))
sn.set(font_scale=1.2) # label and title size
plt.subplot(2, 1 ,1)
plt.title('MLP | Confusion Matrix')
sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 9}, fmt='g') #annot_kws is value font
# fmt='g' --> turnoff the scientific notation
plt.subplot(2, 1 ,2)
plt.title('MLP | Normalized Confusion Matrix')
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


# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# print(TPR)
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# print(TNR)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# print(PPV)
# # Negative predictive value
# NPV = TN/(TN+FN)
# print(NPV)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# print(FPR)
# # False negative rate
# FNR = FN/(TP+FN)
# print(FNR)
# # False discovery rate
# FDR = FP/(TP+FP)
# print(FDR)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print("FP: ", FP)
print("FN: ", FN)
print("TP: ", TP)
print("TN: ", TN)
print("ACC: ", ACC)
