import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Activation, Dropout, GlobalAvgPool2D
import tensorflow as tf
import keras

# The three model
def create_modelTri():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(300, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation='softmax'))
    return model

def create_modelRec():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1))) 
    model.add(Dense(200, activation="relu"))   
    model.add(Dropout(rate=0.5))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(10, activation="softmax"))
    return model

def create_modelHex():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1))) 
    model.add(Dense(175, activation="relu"))   
    model.add(Dropout(rate=0.5))
    model.add(Dense(300, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(175, activation="relu"))
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(10, activation="softmax"))
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
