from keras.datasets import cifar100
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # to visualize data and draw plots
from math import sqrt

def load_data():
    (images, labels), (temp, temp1) = cifar100.load_data()
    train_images, validation_images = images[:40000], images[40000:]
    train_labels, validation_labels = labels[:40000], labels[40000:]

    train_images = train_images.astype('float32')
    validation_images = validation_images.astype('float32')
    images = images.astype('float32')
    temp = temp.astype('float32')
    images /= 255
    train_images /= 255
    validation_images /= 255
    temp /= 255

    train_labels = keras.utils.to_categorical(train_labels)
    validation_labels_save = keras.utils.to_categorical(validation_labels)
    return train_images, train_labels, validation_images, validation_labels, validation_labels_save

def print_confusion_matrix(validation_labels, y_predict):
    y_prediction_bin = np.array([])
    for example in y_predict:
        maxN = example[0]
        ind = 0
        for i in range(1, 100):
            if example[i] > maxN:
                ind = i
                maxN = example[i]
        y_prediction_bin = np.append(y_prediction_bin, ind)

    y_prediction_bin = y_prediction_bin.astype(int)
    confusion_matrix = tf.math.confusion_matrix(validation_labels, y_prediction_bin)
    np.savetxt("HW1_model1_confusion_matrix.txt", confusion_matrix.numpy(), fmt='%03.d')
    return confusion_matrix


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
