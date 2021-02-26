# In[1]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math as m
import matplotlib.pyplot as plt    # to visualize data and draw plots
import numpy as np
import seaborn as sns


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist

(images, labels), (temp, temp1) = fashion_mnist.load_data()

# Partitioned the data 50,000 training data and 10,000 validation data
train_images, validation_images= images[50000:] / 255.0, images[:10000] / 255.0  
train_labels, validation_labels= labels[50000:], labels[:10000]

classes=[0,1,2,3,4,5,6,7,8,9] # To make heat map


# In[1]:
# Graph model
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.xlabel("Epoch")
plt.ylabel("Percentage")
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# In[6]:
# Create confidence interval for error rate on validation data
error_test_data = 1 - evaluation[1]     # 1 - correct predictions to get rate of wrong predictions

# Make upper and lower bound of error on test data assuming error follows a normal distribution
upperbound_gen_error = round(error_test_data + 1.96*(m.sqrt(((error_test_data)*(1 - error_test_data))/len(validation_labels))), 3)  
lowerbound_gen_error = round(error_test_data - 1.96*(m.sqrt(((error_test_data)*(1 - error_test_data))/len(validation_labels))), 3)
print("Based on using normal distribution as the estimator, the 95% confidence interval for error hypothesis: [", lowerbound_gen_error, ",", upperbound_gen_error, "]")


# In[7]:
# Making the confusion matrix based on the website: https://androidkt.com/keras-confusion-matrix-in-tensorboard/
# Feel free to check the website to make any changes to the code. We need to change this since it's exact copy
y_pred=model.predict_classes(validation_images)   # Make prediction
con_Matrix = tf.math.confusion_matrix(labels=validation_labels, predictions=y_pred).numpy()

con_Matrix_norm = np.around(con_Matrix.astype('float') / con_Matrix.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_Matrix_norm,
                     index = classes, 
                     columns = classes)
#plots the heat map of the con matrix
figure = plt.figure()
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()