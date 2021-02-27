import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math as m
import matplotlib.pyplot as plt    # to visualize data and draw plots
import numpy as np
import seaborn as sns

import util_FMNIST #imports our until file
import model_FMNIST #imports our model file


#pulls in data and spilts between train and validatoin (temp)
train_images, train_labels, validation_images, validation_labels = util_FMNIST.load_data()


#uncomment the model taht you want to run
#model = model_FMNIST.create_modelTri()
#model = model_FMNIST.create_modelRec()
model = model_FMNIST.create_modelHex()
model.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = "adam", metrics = ["accuracy"])

history = model.fit(train_images, train_labels, batch_size = 32, epochs=100, validation_data=(validation_images, validation_labels), use_multiprocessing=(True))

validation_evaluation = model.evaluate(validation_images, validation_labels) # Stores [loss, accuracy] of model



#all until functions
y_predict = model.predict(validation_images)
y_pred=model.predict_classes(validation_images)#for confusion matrix

util_FMNIST.print_confusion_matrix(validation_labels,y_pred)
util_FMNIST.print_accuracy_graph(history)
util_FMNIST.print_loss_graph(history)
lower_bound_interval, upper_bound_interval = util_FMNIST.confidence_interval(validation_evaluation, len(validation_labels))
print("The 95% Confidence interval for error hypothesis based on the normal distribution estimator: ",
      (lower_bound_interval, upper_bound_interval))
tf.saved_model.save(model, 'homework1/FMNIST')




