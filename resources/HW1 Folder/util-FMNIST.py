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
train_images, validation_images= images[50000:] / 255.0, images[:10000] / 255.0  
train_labels, validation_labels= labels[50000:], labels[:10000]

evaluation = model.evaluate(validation_images, validation_labels)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.xlabel("Epoch")
plt.ylabel("Percentage")
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Create confidence interval for error rate on validation data
error_test_data = 0
y_pred=model.predict_classes(validation_images)   # Make prediction
for length in range(len(y_pred)):                 # For loop that counts how many correct predictions
    if y_pred[length] == validation_labels[length]:   
        error_test_data +=  1/len(validation_labels)
        
error_test_data = 1 - error_test_data     # 1 - correct predictions to get rate of wrong predictions

# Make upper and lower bound of error on test data assuming error follows a normal distribution
upperbound_gen_error = error_test_data + 1.96*(m.sqrt(((error_test_data)*(1 - error_test_data))/len(validation_labels)))  
lowerbound_gen_error = error_test_data - 1.96*(m.sqrt(((error_test_data)*(1 - error_test_data))/len(validation_labels)))
print("Based on using normal distribution as the estimator, the 95% confidence interval for error hypothesis: [", lowerbound_gen_error, ",", upperbound_gen_error, "]")

# Making the confusion matrix based on the website: https://androidkt.com/keras-confusion-matrix-in-tensorboard/
# Feel free to check the website to make any changes to the code. We need to change this since it's exact copy
con_Matrix = tf.math.confusion_matrix(labels=validation_labels, predictions=y_pred).numpy()

con_Matrix_norm = np.around(con_Matrix.astype('float') / con_Matrix.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_Matrix_norm,
                     index = classes, 
                     columns = classes)
