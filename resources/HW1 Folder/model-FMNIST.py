import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt    # to visualize data and draw plots


model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    keras.layers.Dense(300, activation="relu"),    # Change the 500 based on the shape you want
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(10, activation="softmax"),
    ])

model.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = "adam",
              metrics = ["accuracy"])

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30, restore_best_weights=(True))

history = model.fit(train_images, train_labels, batch_size = 32, epochs=100, 
          validation_data=(validation_images, validation_labels), use_multiprocessing=(True))

