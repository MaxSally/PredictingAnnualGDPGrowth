from sklearn.model_selection import train_test_split
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
from keras.callbacks import EarlyStopping, ModelCheckpoint

from util import *
from model import *

#gets the IMDB data
train_text, train_label = get_train_data()

#sets variables needed for the model
vocab_size = 60000
maxlen = 250
encode_dim = 70
batch_size = 32

#gets the text needed to train the model
train_text_padded = get_padded_train_text(train_text)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#gets the model that we want to use
#change getModelN with (1,2,3 or 4) for the different moddels.
#train_model = getModel3(train_text_padded.shape[1], vocab_size)


#this was used to train and test the overall model
"""
final_train_text, final_validation_text, final_train_label, final_validation_label = train_test_split(train_text_padded, train_label, test_size=0.2, shuffle=True)

final_train_text = np.array(final_train_text)
final_validation_text = np.array(final_validation_text)
final_train_label = np.array(final_train_label)
final_validation_label = np.array(final_validation_label)
history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val), callbacks = [es])
"""

#this was used for k-folds
k = 3 #number of folds
numfold = 0 #what fold we are on

for i in range(k):
    #gets the model that we want to use
    #change getModelN with (1,2,3 or 4) for the different moddels.
    train_model = getModel3(train_text_padded.shape[1], vocab_size)

    print("Fold :", numfold)#prints what fold we are on
    numfold += 1
    lenght = len(train_text_padded) #gets the lenght of the data
    start = i*lenght//k #gets the starting index for each fold
    end = (i+1)*lenght//k #gets the ending index for each fold
    
    val_padded_fold = train_text_padded[start:end]#splits the data into a k size pice to validatoin
    
    train_padded_fold = list(train_text_padded[:start])
    for e in train_text_padded[end:]:#splits the data into a everything but the k size pice to train
        train_padded_fold.append(e)
        
        
    val_label_2_fold = train_label[start:end]#splits the data labes into a k size pice to validatoin
    train_label_2_fold = list(train_label[:start])
    for e in train_label[end:]:
        train_label_2_fold.append(e)
    
    
    
    final_train_text = np.array(train_padded_fold)
    final_train_label = np.array(train_label_2_fold)
    
    final_validation_text = np.array(val_padded_fold)
    final_validation_label = np.array(val_label_2_fold)
    
    history = train_model.fit(final_train_text, final_train_label, epochs=50, batch_size=batch_size, verbose=1, validation_data=(final_validation_text, final_validation_label), callbacks=[es])
    #saveModel(train_model, 'homework2/imdb')

    validation_evaluation = train_model.evaluate(final_validation_text, final_validation_label)

    validation_predict = train_model.predict(final_validation_text)

    confusion_matrix = print_confusion_matrix(final_validation_label, validation_predict)

    print(confusion_matrix)

    print_accuracy_graph(history)

    print_loss_graph(history)

    (lower_bound_interval, upper_bound_interval) = confidence_interval(validation_evaluation, len(final_validation_label))

    print("The 95% Confidence interval for error hypothesis based on the normal distribution estimator: ",
          (lower_bound_interval, upper_bound_interval))
