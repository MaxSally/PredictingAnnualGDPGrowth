#!/usr/bin/env python
# coding: utf-8

# In[2]:


# We'll start with our library imports...
from __future__ import print_function
from tensorflow import keras 
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from tensorflow.keras.datasets import imdb


# In[13]:


# load the text dataset
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:80%]', 'train[20%:]', 'test'),
    as_supervised=True)
x_input = train_data.batch(1024)
x_validation = validation_data.batch(1024)
y_test_data = test_data.batch(1024)
MAX_SEQ_LEN = 200
MAX_TOKENS = 5000


# In[14]:


print(x_input)


# In[22]:


inputLayer = tf.keras.layers.Input(shape = (), dtype = tf.string, name = "input")
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens = MAX_TOKENS, 
    output_mode='int',
    output_sequence_length = MAX_SEQ_LEN)

embedding_layer = tf.keras.layers.Embedding(input_dim = 5000, 
                                            output_dim = 256, 
                                            mask_zero = True)

BIDI1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True))
GlobalPool = tf.keras.layers.GlobalMaxPooling1D()
LSTM1 = tf.keras.layers.LSTM(512, return_sequences = True)
LSTM2 = tf.keras.layers.LSTM(512, return_sequences = True)
gru1 = tf.keras.layers.GRU(256, return_sequences = True) 
gru2 = tf.keras.layers.GRU(256)
layer128 = tf.keras.layers.Dense(128, activation="relu")

layer64 = tf.keras.layers.Dense(64, activation="relu")

layer32 = tf.keras.layers.Dense(32, activation="relu")

output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

model1 = tf.keras.Sequential([
    inputLayer, 
    vectorize_layer, 
    embedding_layer,
    BIDI1,
    gru1,
    GlobalPool,
    layer128,   
    output_layer
])

model1.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[23]:


history = model1.fit(x = x_input,
                    epochs=50,
                    validation_data=x_validation,
                    verbose=1)


# In[38]:


results = model1.evaluate(y_test_data, verbose=2)


# In[37]:


tf.saved_model.save(model1, '/work/cse479/izzatadly/HW2Model/Comp1IMDB')


# In[ ]:




