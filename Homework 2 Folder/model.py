# We'll start with our library imports...
from __future__ import print_function
from tensorflow import keras 
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Activation, Dropout, GlobalAvgPool2D

# The three model

def create_model1(training_data):
    MAX_SEQ_LEN = 200
    MAX_TOKENS = 5000
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                                                                max_tokens = MAX_TOKENS, 
                                                                output_mode='int',
                                                                output_sequence_length = MAX_SEQ_LEN)
    vectorize_layer.adapt(training_data.map(lambda text, label: text))
    model = Sequential()
    model.add(tf.keras.layers.Input(shape = (), dtype = tf.string, name = "input"))
    model.add(tf.keras.layers.Embedding(input_dim = 5000, 
                                            output_dim = 512, 
                                            mask_zero = True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)))
    model.add(tf.keras.layers.GRU(256, return_sequences = True))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(Dense(tf.keras.layers.Dense(128, activation="relu")))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model


# In[4]:

"""
model = hexagon # Change based on the model you want to test
model.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = "adam", metrics = ["accuracy"])
history = model.fit(train_images, train_labels, batch_size = 32, epochs=100, 
          validation_data=(validation_images, validation_labels), use_multiprocessing=(True))
evaluation = model.evaluate(validation_images, validation_labels) # Stores [loss, accuracy] of model
"""