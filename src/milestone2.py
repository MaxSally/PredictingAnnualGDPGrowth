#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from __future__ import print_function
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import keras
import pandas as pd
from keras import backend
from tensorflow.keras import Model, initializers, regularizers, constraints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt 


# In[2]:


inputs = pd.read_csv('input.csv')


# In[3]:


print(inputs)


# In[4]:


keys = []
for it in inputs:
    keys.append(it)


# In[5]:


print(keys)


# In[6]:


distribution = [4, 9, 24, 26, 27, 28] #horrible design

print(keys[distribution[4] - 1])


# In[7]:


number_of_tests = len(inputs[keys[0]])


# In[ ]:





# In[20]:


input_short_term = []
input_long_term = []
output = []
for i in range(number_of_tests):
    instance = []
    if pd.isna(inputs[keys[0]][i]):
        continue
    for j in range(distribution[0], distribution[1]):         
        instance.append(round(float(inputs[keys[j]][i])/100, 2))
        
    input_short_term.append(instance)
    instance = []
    for j in range(distribution[1], distribution[2]):
        instance.append(round(float(inputs[keys[j]][i])/100, 2))
    for j in range(distribution[3], distribution[4]):
        instance.append(round(float(inputs[keys[j]][i])/100, 2))
    input_long_term.append(instance)
    
    output.append(round(float(inputs[keys[distribution[5] - 1]][i]), 2))

# print(training_input_short_term)
# print(training_input_long_term)
# print(output)


# In[21]:


input_short_term = np.array(input_short_term)
input_long_term = np.array(input_long_term)
output = np.array(output)


# In[11]:


actual_number_of_tests = len(input_short_term)


# In[ ]:


model = Sequential()
model.add(Dense(50, activation='relu', input_shape = input_short_term[0].shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='mean_absolute_error')


# In[ ]:


#Fitting the model
history = model.fit(training_input_short_term, output, epochs = 10, batch_size = 32, verbose = 1)


# In[ ]:


model = Sequential()
model.add(Dense(50, activation='relu', input_shape = input_long_term[0].shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='mean_absolute_error')


# In[ ]:


#Fitting the model
history = model.fit(training_input_long_term, output, epochs = 10, batch_size = 32, verbose = 1)


# In[12]:


short_term_input1 = Input(shape=(5,))
short_term_input2 = Input(shape=(5,))
long_term_input = Input(shape=(16,))
dense7 = Dense(50, activation='relu')
x1 = dense7(short_term_input1)
x1 = Dropout(0.5)(x1)
x1 = Dense(50)(x1)
x1 = LeakyReLU(0.2)(x1)

con1 = Concatenate(axis=1)([x1, short_term_input2, long_term_input])
#con1 = LSTM(9, return_sequences = True)(con1)
con1 = Dense(50, activation='relu')(con1)

#x2 = LSTM(3, return_sequences = True)(long_term_input)

x2 = Dense(50, activation='relu')(long_term_input)
con2 = Concatenate(axis=1)([con1,x2] ) 
output = Dense(1, activation=None)(con2)
model = Model(inputs=[short_term_input1, short_term_input2, long_term_input], outputs=output)
model.summary()


# In[13]:


model.compile(optimizer='adam', loss='mean_absolute_error')


# In[14]:


es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 5)


# In[ ]:


#[training_input_short_term, training_input_short_term, training_input_long_term], train_output, [testing_input_short_term, testing_input_short_term, testing_input_long_term], test_output = train_test_split([input_short_term, input_short_term, input_long_term], output, shuffle=True)


# In[23]:


#Fitting the model
history = model.fit([input_short_term, input_short_term, input_long_term], output, epochs = 1000, batch_size = 32, verbose = 1, shuffle=True, callbacks=[es])


# In[ ]:




