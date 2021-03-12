#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from __future__ import print_function
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import keras
from keras import backend
from tensorflow.keras import Model, initializers, regularizers, constraints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, GlobalMaxPooling1D, Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Dropout, GlobalAvgPool2D, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from sklearn.model_selection import KFold 


# In[2]:


train_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train', 'test'),
    as_supervised=False)


# In[3]:


temp = train_data.as_numpy_iterator()
train_data_2 = []
train_label_2 = []
for it in temp:
    train_data_2.append(it['text'])
    train_label_2.append(it['label'])


# In[4]:


temp = test_data.as_numpy_iterator()
test_data_2 = []
test_label_2 = []
for it in temp:
    test_data_2.append(it['text'])
    test_label_2.append(it['label'])


# In[5]:


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
    #return tf.squeeze(tf.keras.layers.dot(inputs = [x, kernel], axes=1))
    #return K.dot(x, kernel)


# In[6]:


vocab_size = 60000
maxlen = 250
encode_dim = 70
batch_size = 32


# In[7]:



tokenizer = Tokenizer()
cnt = 0
cnt_1 = 0
for it in train_data_2:
    if cnt % 1000 == 0:
        #print(str(it))
        cnt_1 += 1
        #print(cnt_1)
    cnt += 1
    tokenizer.fit_on_texts(str(it))


# In[8]:


cnt = 0
tokenized_word_list = []
for it in train_data_2:
    if cnt % 1000 == 0:
        print(len(tokenized_word_list))
    cnt += 1
    tokenized_word_list.append(tokenizer.texts_to_sequences(str(it)))


# In[9]:


tokenized_word_list_2 = []
for it in train_data_2:
    temp = tokenizer.texts_to_sequences(str(it))
    newL = []
    for it2 in temp:
        if it2 == []:
            continue
        newL.append(it2[0])
    tokenized_word_list_2.append(newL)


# In[10]:


X_train_padded = pad_sequences(tokenized_word_list_2, maxlen = maxlen, padding='post')


# In[11]:


print(len(X_train_padded))
print(X_train_padded.shape[1])


# In[12]:


es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('model_best.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)


# In[21]:


#model 1
#k-fold cross validation
k = 3 #number of folds
numfold = 0 #what fold we are on

for i in range(k):
    model = Sequential()
    embed = Embedding(input_dim = vocab_size, output_dim = 20, input_length = X_train_padded.shape[1])

    model.add(embed)
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(200, return_sequences = True)))
    model.add(Dropout(0.3))
    model.add(AttentionWithContext())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #model.summary()

    print("Fold :", numfold)
    numfold += 1
    lenght = len(X_train_padded)
    start = i*lenght//k #gets the starting index for each fold
    end = (i+1)*lenght//k #gets the ending index for each fold
    
    val_padded_fold = X_train_padded[start:end]#splits the data into a k size pice to validatoin
    
    train_padded_fold = list(X_train_padded[:start])
    for e in X_train_padded[end:]:#splits the data into a everything but the k size pice to train
        train_padded_fold.append(e)
        
        
    val_label_2_fold = train_label_2[start:end]#splits the data labes into a k size pice to validatoin
    train_label_2_fold = list(train_label_2[:start])
    for e in train_label_2[end:]:
        train_label_2_fold.append(e)
    print(len(train_padded_fold),len(train_label_2_fold))
    
    
    #X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded_fold, train_label_2_fold, test_size = 0.2, shuffle=True)
    
    X_train_final2 = np.array(train_padded_fold)
    y_train_final2 = np.array(train_label_2_fold)
    
    X_val = np.array(val_padded_fold)
    y_val = np.array(val_label_2_fold)
    
    history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val), callbacks = [es])


# In[14]:


#k-fold cross validation
#model 2
k = 3 #number of folds
numfold = 0 #what fold we are on

for i in range(k):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=X_train_padded.shape[1]))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(AttentionWithContext())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()

    print("Fold :", numfold)
    numfold += 1
    lenght = len(X_train_padded)
    start = i*lenght//k #gets the starting index for each fold
    end = (i+1)*lenght//k #gets the ending index for each fold
    
    val_padded_fold = X_train_padded[start:end]#splits the data into a k size pice to validatoin
    
    train_padded_fold = list(X_train_padded[:start])
    for e in X_train_padded[end:]:#splits the data into a everything but the k size pice to train
        train_padded_fold.append(e)
        
        
    val_label_2_fold = train_label_2[start:end]#splits the data labes into a k size pice to validatoin
    train_label_2_fold = list(train_label_2[:start])
    for e in train_label_2[end:]:
        train_label_2_fold.append(e)
    print(len(train_padded_fold),len(train_label_2_fold))
    
    
    #X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded_fold, train_label_2_fold, test_size = 0.2, shuffle=True)
    
    X_train_final2 = np.array(train_padded_fold)
    y_train_final2 = np.array(train_label_2_fold)
    
    X_val = np.array(val_padded_fold)
    y_val = np.array(val_label_2_fold)
    
    history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val), callbacks = [es])


# In[ ]:


#model 3
#k-fold cross validation
k = 3 #number of folds
numfold = 0 #what fold we are on

for i in range(k):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=X_train_padded.shape[1]))
    model.add(Dropout(0.4))
    model.add(Bidirectional(GRU(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(AttentionWithContext())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("Fold :", numfold)
    numfold += 1
    lenght = len(X_train_padded)
    start = i*lenght//k #gets the starting index for each fold
    end = (i+1)*lenght//k #gets the ending index for each fold
    
    val_padded_fold = X_train_padded[start:end]#splits the data into a k size pice to validatoin
    
    train_padded_fold = list(X_train_padded[:start])
    for e in X_train_padded[end:]:#splits the data into a everything but the k size pice to train
        train_padded_fold.append(e)
        
        
    val_label_2_fold = train_label_2[start:end]#splits the data labes into a k size pice to validatoin
    train_label_2_fold = list(train_label_2[:start])
    for e in train_label_2[end:]:
        train_label_2_fold.append(e)
    print(len(train_padded_fold),len(train_label_2_fold))
    
    
    #X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded_fold, train_label_2_fold, test_size = 0.2, shuffle=True)
    
    X_train_final2 = np.array(train_padded_fold)
    y_train_final2 = np.array(train_label_2_fold)
    
    X_val = np.array(val_padded_fold)
    y_val = np.array(val_label_2_fold)
    
    history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val), callbacks = [es])


# In[ ]:


#k-fold cross validation
#model 4
k = 3 #number of folds
numfold = 0 #what fold we are on

for i in range(k):
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
    #model.summary()

    print("Fold :", numfold)
    numfold += 1
    lenght = len(X_train_padded)
    start = i*lenght//k #gets the starting index for each fold
    end = (i+1)*lenght//k #gets the ending index for each fold
    
    val_padded_fold = X_train_padded[start:end]#splits the data into a k size pice to validatoin
    
    train_padded_fold = list(X_train_padded[:start])
    for e in X_train_padded[end:]:#splits the data into a everything but the k size pice to train
        train_padded_fold.append(e)
        
        
    val_label_2_fold = train_label_2[start:end]#splits the data labes into a k size pice to validatoin
    train_label_2_fold = list(train_label_2[:start])
    for e in train_label_2[end:]:
        train_label_2_fold.append(e)
    print(len(train_padded_fold),len(train_label_2_fold))
    
    
    #X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded_fold, train_label_2_fold, test_size = 0.2, shuffle=True)
    
    X_train_final2 = np.array(train_padded_fold)
    y_train_final2 = np.array(train_label_2_fold)
    
    X_val = np.array(val_padded_fold)
    y_val = np.array(val_label_2_fold)
    
    history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val), callbacks = [es])


# In[18]:


save_model = model


# In[14]:


cnt = 0
for it in test_data_2:
    if cnt % 1000 == 0:
        print(str(it))
        print(cnt //1000)
    cnt += 1
    tokenizer.fit_on_texts(str(it))


# In[15]:


cnt = 0
tokenized_word_list_test = []
for it in test_data_2:
    if cnt % 1000 == 0:
        print(len(tokenized_word_list_test))
    cnt += 1
    tokenized_word_list_test.append(tokenizer.texts_to_sequences(str(it)))


# In[16]:


tokenized_word_list_2_test = []
for it in test_data_2:
    temp = tokenizer.texts_to_sequences(str(it))
    newL = []
    for it2 in temp:
        if it2 == []:
            continue
        newL.append(it2[0])
    tokenized_word_list_2_test.append(newL)


# In[17]:


X_test_padded = pad_sequences(tokenized_word_list_2_test, maxlen = maxlen, padding='post')


# In[18]:


test_model = tf.keras.models.load_model('homework2/imdb')


# In[20]:


test_model.evaluate(np.array(X_test_padded), np.array(test_label_2))


# In[ ]:


model.evaluate(X_test_padded, test_label_2)


# In[ ]:


MAX_SEQ_LEN = 128
MAX_TOKENS = 5000

# load the text dataset


# Create TextVectorization layer
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=MAX_SEQ_LEN)

# Use `adapt` to create a vocabulary mapping words to integers
#train_text = 
vectorize_layer.adapt(train_data.map(lambda x: x['text']))


# In[ ]:


# Let's print out a batch to see what it looks like in text and in integers
for text in train_text:
    text = tf.convert_to_tensor([text], dtype='string')
    print(list(zip(text.numpy(), vectorize_layer(text).numpy())))
    break


# In[ ]:


VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
print("Vocab size is {} and is embedded into {} dimensions".format(VOCAB_SIZE, EMBEDDING_SIZE))

embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)


# In[ ]:


# for batch in validation_data:
#     print(batch)


# In[ ]:


query_input = Input(shape=(None,), dtype='float')
value_input = Input(shape=(None,), dtype='float')

# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = embedding_layer(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = embedding_layer(value_input)

# CNN layer.
cnn_layer = Conv1D(
    filters=100,
    kernel_size=4,
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)
print(query_value_attention)
# Concatenate query and document encodings to produce a DNN input layer.
input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])


# In[ ]:


print(type(train_data_2[0]))


# In[ ]:


BIDI1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True))
GlobalPool = tf.keras.layers.GlobalMaxPooling1D()
LSTM1 = tf.keras.layers.LSTM(512, return_sequences = True)
LSTM2 = tf.keras.layers.LSTM(256, return_sequences = True)
gru1 = tf.keras.layers.GRU(256, return_sequences = True) 
gru2 = tf.keras.layers.GRU(256)

# input_layer = BIDI1(input_layer)
# input_layer = GlobalPool(input_layer)
# input_layer = tf.keras.layers.LSTM(512)(input_layer)
# input_layer = tf.keras.layers.LSTM(256)(input_layer)

output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

model = keras.Model(inputs=input_layer, outputs=output_layer, name="crying_model")
model.summary()


# In[ ]:


# We'll make a conv layer to produce the query and value tensors
query_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    padding='same')
value_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    padding='same')
# Then they will be input to the Attention layer
attention = tf.keras.layers.Attention()
concat = tf.keras.layers.Concatenate()

cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
rnn = tf.keras.layers.RNN(cells)
output_layer = tf.keras.layers.Dense(1)

cnt = 0
loss_values = []
for epoch in range(5):
    loss_values_per_epoch = []
    for batch in train_data.batch(32):
        text = batch['text']
        embeddings = embedding_layer(vectorize_layer(text))
        query = query_layer(embeddings)
        value = value_layer(embeddings)
        query_value_attention = attention([query, value])
        #print("Shape after attention is (batch, seq, filters):", query_value_attention.shape)
        attended_values = concat([query, query_value_attention])
        #print("Shape after concatenating is (batch, seq, filters):", attended_values.shape)
        logits = output_layer(rnn(attended_values))
        loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(batch['label'], -1), logits, from_logits=True)
        loss_values_per_epoch.append(loss)
    
print(tf.reduce_mean(loss_value))


# In[ ]:




