from tensorflow import keras

# Helper libraries
#import numpy as np
#import matplotlib.pyplot as plt
import tensorflow_datasets as tfds # to load training data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt    # to visualize data and draw plots

fashion_mnist = keras.datasets.fashion_mnist


(images, labels), (temp, temp1) = fashion_mnist.load_data()
temp = temp /255.0
train_images, validation_images = images[5000:] / 255.0, images[:5000] / 255.0
train_labels, validation_labels = labels[5000:], labels[:5000]


model.evaluate(temp, temp1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

