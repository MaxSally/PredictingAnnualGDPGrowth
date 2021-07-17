from sklearn.model_selection import train_test_split
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
from keras.callbacks import EarlyStopping, ModelCheckpoint

from util import *
from model import *

train_text, train_label = get_train_data()

vocab_size = 60000
maxlen = 250
encode_dim = 70
batch_size = 32

train_text_padded = get_padded_train_text_fromfile(maxlen)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

train_model = getModel3(train_text_padded.shape[1], vocab_size)

#not sure why, but the shuffle is important for the validation data to process accurately.
final_train_text, final_validation_text, final_train_label, final_validation_label = train_test_split(train_text_padded, train_label, test_size=0.2, shuffle=True)

final_train_text = np.array(final_train_text)
final_validation_text = np.array(final_validation_text)
final_train_label = np.array(final_train_label)
final_validation_label = np.array(final_validation_label)

history = train_model.fit(final_train_text, final_train_label, epochs=50, batch_size=batch_size, verbose=1, validation_data=(final_validation_text, final_validation_label), callbacks=[es])

saveModel(train_model, 'homework2/imdb')

validation_evaluation = train_model.evaluate(final_validation_text, final_validation_label)

validation_predict = train_model.predict(final_validation_text)

confusion_matrix = print_confusion_matrix(final_validation_label, validation_predict)

print(confusion_matrix)

print_accuracy_graph(history)

print_loss_graph(history)

(lower_bound_interval, upper_bound_interval) = confidence_interval(validation_evaluation, len(final_validation_label))

print("The 95% Confidence interval for error hypothesis based on the normal distribution estimator: ",
      (lower_bound_interval, upper_bound_interval))