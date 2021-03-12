#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


train_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train', 'test'),
    as_supervised=False)


# In[12]:


temp = train_data.as_numpy_iterator()
train_data_2 = []
train_label_2 = []
for it in temp:
    train_data_2.append(it['text'])
    train_label_2.append(it['label'])


# In[5]:


temp = test_data.as_numpy_iterator()
test_data_2 = []
test_label_2 = []
for it in temp:
    test_data_2.append(it['text'])
    test_label_2.append(it['label'])


# In[5]:



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


# In[9]:


vocab_size = 60000
maxlen = 250
encode_dim = 70
batch_size = 32


# In[13]:



tokenizer = Tokenizer()
cnt = 0
cnt_1 = 0
for it in train_data_2:
    if cnt % 1000 == 0:
        print(str(it))
        cnt_1 += 1
        print(cnt_1)
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


# In[13]:


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
model.summary()


# In[14]:


X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded, train_label_2, test_size = 0.2, shuffle=True)


# In[15]:


X_train_final2 = np.array(X_train_final2)
X_val = np.array(X_val)
y_train_final2 = np.array(y_train_final2)
y_val = np.array(y_val)


# In[16]:


#Fitting the model
history = model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = (X_val, y_val), callbacks = [es])


# In[29]:


print(history)


# In[17]:


tf.saved_model.save(model, 'homework2/imdb')


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




