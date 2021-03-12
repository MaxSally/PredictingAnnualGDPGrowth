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

#import model #imports our model file

def create_model1():
    model.add(tf.keras.layers.Input(shape = (), dtype = tf.string, name = "input"))
    model.add(vectorize_layer)
    model.add(tf.keras.layers.Embedding(input_dim = 5000, 
                                            output_dim = 512, 
                                            mask_zero = True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)))
    model.add(tf.keras.layers.GRU(256, return_sequences = True))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

#pulls in data and spilts between train and validatoin (temp)
# load the text dataset
train_data, validation_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:80%]', 'train[20%:]'),
    as_supervised=True)
x_input = train_data.batch(1024)
x_validation = validation_data.batch(1024)

# Begin Model
model = Sequential()

# Create vectorize_layer independently
MAX_SEQ_LEN = 200
MAX_TOKENS = 5000
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                                                                max_tokens = MAX_TOKENS, 
                                                                output_mode='int',
                                                                output_sequence_length = MAX_SEQ_LEN)
vectorize_layer.adapt(train_data.map(lambda text, label: text))

create_model1() # Initiate model type 

# create_model2()
model.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = "adam", metrics = ["accuracy"])

history = model.fit(x_input,
                    epochs=40,
                    validation_data = x_validation,
                    verbose=1,
                    use_multiprocessing=(True))