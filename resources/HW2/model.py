import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
from keras import backend
from tensorflow.keras import Model, initializers, regularizers, constraints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, GlobalMaxPooling1D, Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Dropout, GlobalAvgPool2D, Input
from keras.layers import *
from util import *

class Attention(tf.keras.layers.Layer):
    """
    Reference [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    Also https://www.kaggle.com/amanbhalla/imdb-review-lstm-with-attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
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
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='name',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='name',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='name',
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = tf.math.tanh(uit)
        ait = dot_product(uit, self.u)

        a = tf.math.exp(ait)

        if mask is not None:
            a *= tf.cast(mask, tf.float32)

        # epsilon is here to combat the small initial value. Avoid 0 in early training.
        a /= tf.cast(tf.math.reduce_sum(a, axis=1, keepdims=True) + backend.epsilon(), tf.float32)

        a = tf.expand_dims(a, -1)
        print(x.shape)
        print(a.shape)
        weighted_input = x * a
        return tf.math.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def getModel1(input_length, vocab_size=60000):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=input_length))
    model.add(Dropout(0.4))
    model.add(Bidirectional(GRU(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def getModel2(input_length, vocab_size=60000):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=input_length))
    model.add(Dropout(0.4))
    model.add(Bidirectional(GRU(50, return_sequences=True)))
    model.add(Bidirectional(GRU(100, return_sequences=True)))
    model.add(Bidirectional(GRU(50, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def getModel3(input_length, vocab_size=60000):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=input_length))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def getModel4(input_length, vocab_size=60000):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=input_length))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def saveModel(model, filepath):
    tf.saved_model.save(model, filepath)
