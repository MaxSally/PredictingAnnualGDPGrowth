#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We'll start with our library imports...
from __future__ import print_function
import pandas
import seaborn as sn
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from keras.datasets import cifar100
from keras import backend as k 
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout


# In[2]:


(images, labels), (temp, temp1) = cifar100.load_data()
temp = temp /255.0
train_images, validation_images = images[45000:], images[:45000]
train_labels, validation_labels = labels[45000:], labels[:45000]


# In[3]:


print(train_labels)


# In[4]:


img_rows, img_cols, img_width = 32, 32, 3
  
# if k.image_data_format() == 'channels_first': 
#    train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols, img_width) 
#    validation_images = validation_images.reshape(validation_images.shape[0], 1, img_rows, img_cols, img_width) 
#    inpx = (img_rows, img_cols, img_width)
  
# else: 
#    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_width) 
#    validation_images = validation_images.reshape(validation_images.shape[0], img_rows, img_cols, img_width) 
#    inpx = (img_rows, img_cols, img_width) 
  
train_images = train_images.astype('float32') 
validation_images = validation_images.astype('float32') 
images = images.astype('float32')
images /= 255
train_images /= 255
validation_images /= 255
temp /= 255


# In[5]:


train_labels = keras.utils.to_categorical(train_labels)
print(train_labels.shape)
validation_labels = keras.utils.to_categorical(validation_labels)
temp1 = keras.utils.to_categorical(temp1)
labels = keras.utils.to_categorical(labels)


# In[6]:


input_shape = (img_rows, img_cols, img_width)


# In[9]:


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
        input_shape=images.shape[1:], activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(512, (3, 3), padding='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(2048, activation='elu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='softmax'))
model.summary()


# In[10]:


# using Sequential groups all the layers to run at once

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)
history = model.fit(train_images, train_labels, batch_size = 32, epochs=100, validation_data=(validation_images, validation_labels), callbacks = [es], shuffle=True, use_multiprocessing=(True))


# In[11]:


model.evaluate(validation_images, validation_labels)


# In[12]:


model.evaluate(temp, temp1)


# In[13]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[14]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


tf.saved_model.save(model, 'homework1/cifar100-brute')


# In[ ]:




