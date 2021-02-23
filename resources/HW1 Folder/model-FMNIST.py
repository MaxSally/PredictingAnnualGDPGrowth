import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt    # to visualize data and draw plots


invertedTriangle = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    keras.layers.Dense(300, activation="relu"), 
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(10, activation="softmax"),
    ])

rectangle = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    keras.layers.Dense(200, activation="relu"),    
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(10, activation="softmax"),
    ])

hexagon = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    keras.layers.Dense(175, activation="relu"),    
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(175, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(10, activation="softmax"),
    ])

model = invertedTriangle # Change based on the model 
model.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = "adam",
              metrics = ["accuracy"])

history = model.fit(train_images, train_labels, batch_size = 32, epochs=100, 
          validation_data=(validation_images, validation_labels), use_multiprocessing=(True))

