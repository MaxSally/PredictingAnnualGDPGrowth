#!/usr/bin/env python
# coding: utf-8

# In[4]:


# We'll start with our library imports...
import pandas
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from keras.datasets import cifar100
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout
from sklearn.utils import shuffle


# In[5]:


train = tfds.load('cifar100', shuffle_files=True, split='train[:70%]', as_supervised=True)
validation = tfds.load('cifar100', shuffle_files=True, split='train[30%:]', as_supervised=True)


# In[6]:


for example in train:
    print(example)
    print(example[0].numpy())
    print(example[1].numpy())
    break


# In[7]:


(images, labels), (temp, temp1) = cifar100.load_data()
train_images, validation_images = images[:40000], images[40000:]
train_labels, validation_labels = labels[:40000], labels[40000:]


# In[8]:


img_rows, img_cols, img_width = 32, 32, 3

  
train_images = train_images.astype('float32') 
validation_images = validation_images.astype('float32') 
images = images.astype('float32')
temp = temp.astype('float32')
images /= 255
train_images /= 255
validation_images /= 255
temp /= 255


# In[9]:


print(train_labels.shape)
train_labels = keras.utils.to_categorical(train_labels)
print(train_labels.shape)
validation_labels_save = keras.utils.to_categorical(validation_labels)
temp1 = keras.utils.to_categorical(temp1)
#labels = keras.utils.to_categorical(labels)


# In[10]:


input_shape = (img_rows, img_cols, img_width)


# In[11]:


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_images.shape[1:], activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))
model.summary()


# In[12]:


# using Sequential groups all the layers to run at once

optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30)
history = model.fit(train_images, train_labels, batch_size = 32, epochs=1000, validation_data=(validation_images, validation_labels_save), callbacks = [es], shuffle=True, use_multiprocessing=(True))


# In[14]:


validation_evaluation = model.evaluate(validation_images, validation_labels_save)


# In[15]:


y_predict = model.predict(validation_images)


# In[16]:


y_prediction_bin = np.array([])

for example in y_predict:
    maxN = example[0]
    ind = 0
    for i in range(1, 100):
        if example[i] > maxN:
            ind = i
            maxN = example[i]
    y_prediction_bin = np.append(y_prediction_bin, ind)

y_prediction_bin = y_prediction_bin.astype(int)
print(y_prediction_bin) 


# In[17]:


confusion_matrix = tf.math.confusion_matrix(validation_labels, y_prediction_bin)
print(confusion_matrix)


# In[18]:


np.savetxt("HW1_brute_confusion_matrix.txt", confusion_matrix.numpy(), fmt='%03.d')


# In[21]:


model.evaluate(temp, temp1)


# In[22]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[23]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[32]:


from math import sqrt

validation_data_error = 1 - validation_evaluation[1]
print(validation_data_error)


# In[33]:



lower_bound_interval = validation_data_error - 1.96 * sqrt( (validation_data_error * (1 - validation_data_error)) / len(validation_labels))
upper_bound_interval = validation_data_error + 1.96 * sqrt( (validation_data_error * (1 - validation_data_error)) / len(validation_labels))

print("The 95% Confidence interval for error hypothesis based on the normal distribution estimator: ", (lower_bound_interval, upper_bound_interval))


# In[31]:


tf.saved_model.save(model, 'homework1/cifar100-brute')


# In[ ]:




