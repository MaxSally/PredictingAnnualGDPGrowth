import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math as m
import matplotlib.pyplot as plt    # to visualize data and draw plots
import numpy as np
import seaborn as sns
from math import sqrt


classes=[0,1,2,3,4,5,6,7,8,9] # To make heat map

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (images, labels), (temp, temp1) = fashion_mnist.load_data()

    # Partitioned the data 50,000 training data and 10,000 validation data
    train_images, validation_images= images[:50000] / 255.0, images[50000:] / 255.0  
    train_labels, validation_labels= labels[:50000], labels[50000:]
    return train_images, train_labels, validation_images, validation_labels




def print_confusion_matrix(validation_labels, y_pred):
    con_Matrix = tf.math.confusion_matrix(labels=validation_labels, predictions=y_pred).numpy()

    con_Matrix_norm = np.around(con_Matrix.astype('float') / con_Matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_Matrix_norm, index = classes, columns = classes)


    #plots the heat map of the con matrix
    figure = plt.figure()
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def confidence_interval(validation_evaluation, validation_size):
    validation_data_error = 1 - validation_evaluation[1]
    print(validation_data_error)

    lower_bound_interval = validation_data_error - 1.96 * sqrt(
        (validation_data_error * (1 - validation_data_error)) / validation_size)
    upper_bound_interval = validation_data_error + 1.96 * sqrt(
        (validation_data_error * (1 - validation_data_error)) / validation_size)
    return (lower_bound_interval, upper_bound_interval)

def print_accuracy_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def print_loss_graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



