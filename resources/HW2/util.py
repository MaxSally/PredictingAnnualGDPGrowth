import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from math import sqrt

def get_train_data():
    train_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train', 'test'),
        as_supervised=False)
    train_text = []
    train_label = []
    for train_instance in train_data.as_numpy_iterator():
        train_text.append(train_instance['text'])
        train_label.append(train_instance['label'])
    return train_text, train_label


def dot_product(x, kernel):
    return tf.tensordot(x, kernel, axes=1)

def get_padded_train_text(train_text, maxlen = 250):
    tokenizer = Tokenizer()
    cnt = 0
    for it in train_text:
        if cnt % 1000 == 0:
            print(str(it))
            print(cnt//1000)
        cnt += 1
        tokenizer.fit_on_texts(str(it))
    cnt = 0
    tokenized_word_list = []
    for it in train_text:
        if cnt % 1000 == 0:
            print(len(tokenized_word_list))
        cnt += 1
        tokenized_word_list.append(tokenizer.texts_to_sequences(str(it)))

    #The shape of the tokenized_word_list is [, , ,]. This transforms it to [,]
    tokenized_word_list_2 = []
    for it in train_text:
        temp = tokenizer.texts_to_sequences(str(it))
        newL = []
        for it2 in temp:
            if it2 == []:
                continue
            newL.append(it2[0])
        tokenized_word_list_2.append(newL)

    text_train_padded = pad_sequences(tokenized_word_list_2, maxlen=maxlen, padding='post')
    return text_train_padded


def print_confusion_matrix(validation_labels, y_predict):
    y_prediction_bin = np.array([])
    for i in range(len(y_predict)):
        temp = 1 if y_predict[i][0] >= 0.5 else 0
        y_prediction_bin = np.append(y_prediction_bin, temp)
    y_prediction_bin = y_prediction_bin.astype(int)
    confusion_matrix = tf.math.confusion_matrix(validation_labels, y_prediction_bin)
    np.savetxt("HW1_model1_confusion_matrix.txt", confusion_matrix.numpy(), fmt='%03.d')
    return confusion_matrix

def confidence_interval(validation_evaluation, validation_size):
    validation_data_error = 1 - validation_evaluation[1]
    print(validation_data_error)

    lower_bound_interval = validation_evaluation[1] - 1.96 * sqrt(
        (validation_data_error * (1 - validation_data_error)) / validation_size)
    upper_bound_interval = validation_evaluation[1] + 1.96 * sqrt(
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
