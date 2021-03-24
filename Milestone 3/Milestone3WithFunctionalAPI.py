#!/usr/bin/env python
# coding: utf-8

# In[457]:


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
from tensorflow.keras.layers import LSTM, GRU, Reshape, Bidirectional, Dense, Input, Dropout, LeakyReLU, Concatenate, PReLU, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt 


# In[458]:


inputs = pd.read_csv('input.csv')

keys = []
for it in inputs:
    keys.append(it)
    
distribution = [4, 9, 24, 26, 27, 28] #horrible design

number_of_tests = len(inputs[keys[0]])


# In[459]:


input_short_term = []
input_long_term = []
output = []

for i in range(number_of_tests):
    instance = []
    if pd.isna(inputs[keys[0]][i]):
        continue
    for j in range(distribution[0], distribution[1]):         
        instance.append(float(inputs[keys[j]][i])/100)

    input_short_term.append(instance)
    instance = []
    for j in range(distribution[1], distribution[2]):
        instance.append(float(inputs[keys[j]][i])/100)
    for j in range(distribution[3], distribution[4]):
        instance.append(float(inputs[keys[j]][i])/100)
    input_long_term.append(instance)

    output.append(float(inputs[keys[distribution[5] - 1]][i])/100)


# In[460]:


print(len(input_short_term))


# # Appending both short term and long term variables into training and testing set

# In[461]:


training_input_short_term = []
training_input_long_term = []
training_output = []

testing_input_short_term = []
testing_input_long_term = []
testing_output = []

for i in range(8):
    for j in range(11):
        training_input_short_term.append(input_short_term[j])
        training_input_long_term.append(input_long_term[j])
        training_output.append(output[j])
        
        
    del input_short_term[0:11]
    del input_long_term[0:11]
    del output[0:11]
        #training_input_short_term = np.delete(input_short_term, j)
        
    for k in range(3):
        testing_input_short_term.append(input_short_term[k])
        testing_input_long_term.append(input_long_term[k])
        testing_output.append(output[k])
        
    del input_short_term[0:3]
    del input_long_term[0:3]
    del output[0:3]


# In[462]:


print(len(training_input_short_term))


# # Verification that training and testing data set are correct length

# In[463]:


# Verification 
print(len(training_input_short_term))
print(len(training_input_long_term))
print(len(training_input_long_term))

print(len(testing_input_short_term))
print(len(testing_input_long_term))
print(len(testing_output))


# # Creating each specific long term variables training set

# In[464]:


Capital_Investment_train = []
Labor_Force_Participation_train = []
Fixed_Broadband_train = []
RandD_train = []
Property_Rights_train = []
Freedom_From_Corruption_train = []
Fiscal_Freedom_train = []
Business_Freedom_train = []
Labor_Freedom_train = []
Monetary_Freedom_train = []
Trade_Freedom_train = []
Investment_Freedom_train = []
Financial_Freedom_train = []
Economic_Freedom_Overall_train = []
Pop_Above_65_train = []
Savings_As_GDP_train = []


for i in range(len(training_input_long_term)):
    Capital_Investment_train.append(training_input_long_term[i][0])
    Labor_Force_Participation_train.append(training_input_long_term[i][1])
    Fixed_Broadband_train.append(training_input_long_term[i][2])
    RandD_train.append(training_input_long_term[i][3])
    Property_Rights_train.append(training_input_long_term[i][4])
    Freedom_From_Corruption_train.append(training_input_long_term[i][5])
    Fiscal_Freedom_train.append(training_input_long_term[i][6])
    Business_Freedom_train.append(training_input_long_term[i][7])
    Labor_Freedom_train.append(training_input_long_term[i][8])
    Monetary_Freedom_train.append(training_input_long_term[i][9])
    Trade_Freedom_train.append(training_input_long_term[i][10])
    Investment_Freedom_train.append(training_input_long_term[i][11])
    Financial_Freedom_train.append(training_input_long_term[i][12])
    Economic_Freedom_Overall_train.append(training_input_long_term[i][13])
    Pop_Above_65_train.append(training_input_long_term[i][14])
    Savings_As_GDP_train.append(training_input_long_term[i][15])
  


# # Creating each specific long term variable test set

# In[465]:


Capital_Investment_test = []
Labor_Force_Participation_test  = []
Fixed_Broadband_test  = []
RandD_test  = []
Property_Rights_test  = []
Freedom_From_Corruption_test  = []
Fiscal_Freedom_test  = []
Business_Freedom_test  = []
Labor_Freedom_test  = []
Monetary_Freedom_test  = []
Trade_Freedom_test  = []
Investment_Freedom_test  = []
Financial_Freedom_test  = []
Economic_Freedom_Overall_test  = []
Pop_Above_65_test  = []
Savings_As_GDP_test  = []

    
for i in range(len(testing_input_long_term)):
    Capital_Investment_test.append(testing_input_long_term[i][0])
    Labor_Force_Participation_test.append(testing_input_long_term[i][1])
    Fixed_Broadband_test.append(testing_input_long_term[i][2])
    RandD_test.append(testing_input_long_term[i][3])
    Property_Rights_test.append(testing_input_long_term[i][4])
    Freedom_From_Corruption_test.append(testing_input_long_term[i][5])
    Fiscal_Freedom_test.append(testing_input_long_term[i][6])
    Business_Freedom_test.append(testing_input_long_term[i][7])
    Labor_Freedom_test.append(testing_input_long_term[i][8])
    Monetary_Freedom_test.append(testing_input_long_term[i][9])
    Trade_Freedom_test.append(testing_input_long_term[i][10])
    Investment_Freedom_test.append(testing_input_long_term[i][11])
    Financial_Freedom_test.append(testing_input_long_term[i][12])
    Economic_Freedom_Overall_test.append(testing_input_long_term[i][13])
    Pop_Above_65_test.append(testing_input_long_term[i][14])
    Savings_As_GDP_test.append(testing_input_long_term[i][15])  


# # Overall training and testing arrays

# In[466]:


training_input_short_term = np.array(training_input_short_term)
training_input_long_term = np.array(training_input_long_term)
training_output = np.array(training_output)


# In[467]:


testing_input_short_term = np.array(testing_input_short_term)
testing_input_long_term = np.array(testing_input_long_term)
testing_output = np.array(testing_output)


# # Each specific array of long term variables training and test set

# In[468]:


Capital_Investment_train = np.array(Capital_Investment_train)
Capital_Investment_test = np.array(Capital_Investment_test)

Labor_Force_Participation_train = np.array(Labor_Force_Participation_train)
Labor_Force_Participation_test  = np.array(Labor_Force_Participation_test)

Fixed_Broadband_train = np.array(Fixed_Broadband_train)
Fixed_Broadband_test  = np.array(Fixed_Broadband_test)

RandD_train = np.array(RandD_train)
RandD_test  = np.array(RandD_test)

Property_Rights_train = np.array(Property_Rights_train)
Property_Rights_test = np.array(Property_Rights_test)

Freedom_From_Corruption_train = np.array(Freedom_From_Corruption_train)
Freedom_From_Corruption_test  = np.array(Freedom_From_Corruption_test)

Fiscal_Freedom_train = np.array(Fiscal_Freedom_train)
Fiscal_Freedom_test  = np.array(Fiscal_Freedom_test)

Business_Freedom_train = np.array(Business_Freedom_train)
Business_Freedom_test  = np.array(Business_Freedom_test)

Labor_Freedom_train = np.array(Labor_Freedom_train)
Labor_Freedom_test  = np.array(Labor_Freedom_test)

Monetary_Freedom_train = np.array(Monetary_Freedom_train)
Monetary_Freedom_test  = np.array(Monetary_Freedom_test)

Trade_Freedom_train = np.array(Trade_Freedom_train)
Trade_Freedom_test  = np.array(Trade_Freedom_test)

Investment_Freedom_train = np.array(Investment_Freedom_train)
Investment_Freedom_test  = np.array(Investment_Freedom_test)

Financial_Freedom_train = np.array(Financial_Freedom_train)
Financial_Freedom_test  = np.array(Financial_Freedom_test)

Economic_Freedom_Overall_train = np.array(Economic_Freedom_Overall_train)
Economic_Freedom_Overall_test  = np.array(Economic_Freedom_Overall_test)

Pop_Above_65_train = np.array(Pop_Above_65_train)
Pop_Above_65_test  = np.array(Pop_Above_65_test)

Savings_As_GDP_train = np.array(Savings_As_GDP_train)
Savings_As_GDP_test  = np.array(Savings_As_GDP_test)


# In[503]:


print(len(Capital_Investment_test))


# # Setting the long term to be put into a specific model

# 

# In[514]:


# We need to do each country with each long term variable manually to get the randomized complete block design
##  Train  Test
# [0:11] [0:3] - Canada
# [11:22] [3:6] - Greece
# [22:33] [6:9] - Japan
# [33:44] [9:12] - Malaysia
# [44:55] [12:15] - South Africa
# [55:66] [15:18] - Spain
# [66:77] [18:21] - UK
# [77:88] [21:24] - USA
# We need to do each country with each long term variable manually to get the randomized complete block design
## #  Train  Test
# [0:11] [0:3] - Canada
# [11:22] [3:6] - Greece
# [22:33] [6:9] - Japan
# [33:44] [9:12] - Malaysia
# [44:55] [12:15] - South Africa
# [55:66] [15:18] - Spain
# [66:77] [18:21] - UK
# [77:88] [21:24] - USA

trainLV = [Capital_Investment_train, Labor_Force_Participation_train, Fixed_Broadband_train,
          RandD_train, Property_Rights_train, Freedom_From_Corruption_train, Fiscal_Freedom_train, Business_Freedom_train,
          Labor_Freedom_train, Monetary_Freedom_train, Trade_Freedom_train, Investment_Freedom_train, Financial_Freedom_train,
          Economic_Freedom_Overall_train, Pop_Above_65_train, Savings_As_GDP_train]

testLV = [Capital_Investment_test, Labor_Force_Participation_test, Fixed_Broadband_test,
          RandD_test, Property_Rights_test, Freedom_From_Corruption_test, Fiscal_Freedom_test, Business_Freedom_test,
          Labor_Freedom_train, Monetary_Freedom_test, Trade_Freedom_test, Investment_Freedom_test, Financial_Freedom_test,
          Economic_Freedom_Overall_test, Pop_Above_65_train, Savings_As_GDP_test]

# Designated code: Capital Invest = 0
#                  Labor Force = 1
#                  Fixed Broadband = 2
#                  R&D = 3
#                  Property rights = 4
#                  Free from corruption = 5
#                  Fiscal Freedom = 6
#                  Business Freedom = 7
#                  Labor Freedom = 8
#                  Monetary Freedom = 9
#                  Trade Freedom = 10
#                  Investment Freedom = 11
#                  Financial Freedom = 12
#                  Economic Freedom = 13
#                  Pop above 65 = 14
#                  Savings as gdp = 15

# Step 1, only the trainLV and testLV designated code will change. 
# This is because we are the training for the same country but with different long term variables to get the RMSE.
# Hence, only the slicing for the long term variable changes. 

# When we finish all long term variables for one country:
# Step 2, change the slicing to a new country and repeat step 1. Reset the designated code back to 0 from 14.

training_input_short_term_slice = training_input_short_term[0:11] # 0:11 refers to slicing for country training data
training_input_long_term1 = trainLV[3][0:11]   # The 3 is the designated code
training_output_slice = training_output[0:11] 

testing_input_short_term_slice = testing_input_short_term[0:3] # 0:3 refers to slicing for country test data
testing_input_long_term1 = testLV[3][0:3] 
testing_output_slice = training_output[0:3] 

#training_input_long_term2 = Capital_Investment_train
#testing_input_long_term2 = Capital_Investment_test


# #  Single long term variable model (Accurate to the project plan from milestone 2)

# In[516]:


short_term_input1 = Input(shape=(5,))
short_term_input2 = Input(shape=(5,))
long_term_input1 = Input(shape=(1,))

x1 = Dense(50)(short_term_input1)
x1 = LeakyReLU(0.2)(x1)
x1 = Dropout(0.35)(x1)

x1 = Dense(50)(x1)
x1 = LeakyReLU(0.2)(x1)
x1 = Dropout(0.5)(x1)

x1 = Dense(50)(x1)
x1 = LeakyReLU(0.2)(x1)


con1 = Concatenate(axis=1)([x1, short_term_input2, long_term_input1])
con1 = Dropout(0.35)(con1)

con1 = Dense(56, activation = "relu")(con1)
con1 = Dropout(0.2)(con1)

con1 = Reshape(target_shape = (1, 56))(con1)

con1 = LSTM(100, return_sequences = True)(con1)
con1 = Dropout(0.2)(con1)

con1 = LSTM(100, return_sequences = True)(con1)
con1 = Dropout(0.35)(con1)

con1 = LSTM(200, return_sequences = True)(con1)
con1 = Dropout(0.35)(con1)
con1 = LSTM(100, return_sequences = True)(con1)
con1 = LSTM(1, return_sequences = True)(con1)

x2 = Dense(1, activation = None)(long_term_input1)
x2 = Reshape(target_shape= (1, 1))(x2)
x2 = LSTM(100, return_sequences = True)(x2)
x2 = LSTM(1, return_sequences = True)(x2)

con2 = Concatenate(axis=1)([con1, x2]) 

output = Dense(1, activation = None)(con2)

SingleLongTermmodel = Model(inputs=[short_term_input1, short_term_input2, long_term_input1], outputs=output)
SingleLongTermmodel.summary()


# # Mean absolute error

# In[517]:


SingleLongTermmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae', metrics = [tf.keras.metrics.RootMeanSquaredError()])
history = SingleLongTermmodel.fit([training_input_short_term_slice, training_input_short_term_slice, training_input_long_term1], training_output_slice, epochs = 100, batch_size = 1, verbose=4)
history2 = SingleLongTermmodel.evaluate(x=[testing_input_short_term_slice, testing_input_short_term_slice, testing_input_long_term1], y=testing_output_slice, batch_size=1, verbose=1)


# # Mean squared error
# Mean squared error (Metrics = Root mean Square error)

SingleLongTermmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
history = SingleLongTermmodel.fit([training_input_short_term, training_input_short_term, training_input_long_term1], training_output, epochs = 1000, batch_size = 32, verbose = 1)
history2 = SingleLongTermmodel.evaluate(x=[testing_input_short_term, testing_input_short_term, testing_input_long_term1], y=testing_output, batch_size=32, verbose=1)
# # Multiple long term variable
short_term_input1 = Input(shape=(5,))
short_term_input2 = Input(shape=(5,))
long_term_input1 = Input(shape=(1,))
long_term_input2 = Input(shape=(1, ))
x3 = Reshape(target_shape = (1, 1))(long_term_input2)

x1 = Dense(50)(short_term_input1)
x1 = LeakyReLU(0.2)(x1)
x1 = Dropout(0.35)(x1)

x1 = Dense(50)(x1)
x1 = LeakyReLU(0.2)(x1)
x1 = Dropout(0.5)(x1)

x1 = Dense(50)(x1)
x1 = LeakyReLU(0.2)(x1)


con1 = Concatenate(axis=1)([x1, short_term_input2, long_term_input1, long_term_input2])
con1 = Dropout(0.35)(con1)

con1 = Dense(56, activation = "relu")(con1)
con1 = Dropout(0.2)(con1)

con1 = Reshape(target_shape = (1, 56))(con1)

con1 = LSTM(100, return_sequences = True)(con1)
con1 = Dropout(0.2)(con1)

con1 = LSTM(100, return_sequences = True)(con1)
con1 = Dropout(0.35)(con1)

con1 = LSTM(200, return_sequences = True)(con1)
con1 = Dropout(0.35)(con1)
con1 = LSTM(100, return_sequences = True)(con1)
con1 = LSTM(1, return_sequences = True)(con1)


con2 = Concatenate(axis=1)([long_term_input1, long_term_input2])
x2 = Dense(16, activation = "relu")(con2)
x2 = Reshape(target_shape= (1, 16))(x2)
x2 = LSTM(100, return_sequences = True)(x2)
x2 = LSTM(1, return_sequences = True)(x2)

con2 = Concatenate(axis=1)([con1, x2, x3]) 
#con2 = Reshape(target_shape= (3, ))(con2)
(con2) = Dense(100, activation = 'sigmoid')(con2)
output = Dense(1, activation = None)(con2)

MultipleInputmodel = Model(inputs=[short_term_input1, short_term_input2, long_term_input1, long_term_input2], outputs=output)
MultipleInputmodel.summary()

MultipleInputmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_absolute_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])
#model.compile(optimizer= "adam", loss='mean_absolute_error')history = MultipleInputmodel.fit([training_input_short_term, training_input_short_term, training_input_long_term1, training_input_long_term2], training_output, epochs = 10000, batch_size = 32, verbose = 1)
history2 = model.evaluate(x=[testing_input_short_term, testing_input_short_term, testing_input_long_term1, testing_input_long_term2], y=testing_output, batch_size=32, verbose=1)