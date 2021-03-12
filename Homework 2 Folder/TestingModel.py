from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import Model, initializers, regularizers, constraints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, GlobalMaxPooling1D, Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Dropout, GlobalAvgPool2D, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *

# Import data 
train_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train', 'test'),
    as_supervised=False)

temp = train_data.as_numpy_iterator()
train_data_2 = []
train_label_2 = []
for it in temp:
    train_data_2.append(it['text'])
    train_label_2.append(it['label'])
    
temp = test_data.as_numpy_iterator()
test_data_2 = []
test_label_2 = []
for it in temp:
    test_data_2.append(it['text'])
    test_label_2.append(it['label'])
    
# Tokenize the training data
vocab_size = 60000
maxlen = 250
encode_dim = 70
batch_size = 32

tokenizer = Tokenizer()

# Tokenize training data
cnt = 0
cnt_1 = 0
for it in train_data_2:
    #if cnt % 1000 == 0:
        # print(str(it))
        #cnt_1 += 1
        # print(cnt_1)
    #cnt += 1
    tokenizer.fit_on_texts(str(it))
    
cnt = 0
tokenized_word_list = []
for it in train_data_2:
    #if cnt % 1000 == 0:
        #print(len(tokenized_word_list))
    #cnt += 1
    tokenized_word_list.append(tokenizer.texts_to_sequences(str(it)))
    
cnt = 0
tokenized_word_list = []
for it in train_data_2:
    #if cnt % 1000 == 0:
        #(len(tokenized_word_list))
    #cnt += 1
    tokenized_word_list.append(tokenizer.texts_to_sequences(str(it)))
    
tokenized_word_list_2 = []
for it in train_data_2:
    temp = tokenizer.texts_to_sequences(str(it))
    newL = []
    for it2 in temp:
        if it2 == []:
            continue
        newL.append(it2[0])
    tokenized_word_list_2.append(newL)
    
X_train_padded = pad_sequences(tokenized_word_list_2, maxlen = maxlen, padding='post')

# Attention layer
class AttentionWithContext(tf.keras.layers.Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,                
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name = 'name',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[-1],),
                                     initializer='zero',
                                     name = 'name',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name = 'name',
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = tf.math.tanh(uit)
        ait = dot_product(uit, self.u)

        a = tf.math.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= tf.cast(mask, tf.float32)
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= tf.cast(tf.math.reduce_sum(a, axis=1, keepdims=True) + backend.epsilon(), tf.float32)

        a = tf.expand_dims(a, -1)
        print(x.shape)
        print(a.shape)
        weighted_input = x * a
        return tf.math.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    return tf.tensordot(x, kernel, axes = 1)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('model_best.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model = Sequential()
embed = Embedding(input_dim = vocab_size, output_dim = 20, input_length = X_train_padded.shape[1])

model.add(embed)
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Dropout(0.3))
model.add(AttentionWithContext())
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded, train_label_2, test_size = 0.2, shuffle=True)
X_train_final2 = np.array(X_train_final2)
X_val = np.array(X_val)
y_train_final2 = np.array(y_train_final2)
y_val = np.array(y_val)

#Fitting the model
history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val))