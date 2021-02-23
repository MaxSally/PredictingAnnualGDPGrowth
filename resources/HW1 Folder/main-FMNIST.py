import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt    # to visualize data and draw plots

fashion_mnist = keras.datasets.fashion_mnist

(images, labels), (temp, temp1) = fashion_mnist.load_data()
temp = temp /255.0
train_images, validation_images = images[5000:] / 255.0, images[:5000] / 255.0
train_labels, validation_labels = labels[5000:], labels[:5000]

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

history = model.fit(train_images, train_labels, batch_size = 32, epochs=100, 
          validation_data=(validation_images, validation_labels), use_multiprocessing=(True))

model.evaluate(temp, temp1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
    







