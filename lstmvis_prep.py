import numpy as np
import codecs
#import glob
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import word_tokenize
import h5py
import os
from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed
from keras.models import Model


# DATA

data = codecs.open("data/kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
data = data[:20800]
data = [s.lower() for s in data]

labels = codecs.open("data/kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
labels = labels[:20800]
labels = np.array([int(i) for i in labels])

train, dev, train_lab, dev_lab = train_test_split(data, labels, test_size=0.33, random_state=42)

# PARAMETERS
# vocab_size: number of tokens in vocabulary
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
# num_cells: number of LSTM cells
# num_samples: number of training/testing data samples
# num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
# trainTextsSeq: List of input sequence for each document (A matrix with size num_samples * max_doc_length)
# y_train: vector of document class labels

# using the mean length of documents as max_doc_length for now
max_doc_length = int(np.round(np.mean([len(paragraph) for paragraph in train])))
num_time_steps = max_doc_length
# num_cells: number of LSTM cells
num_cells = 100 # 100 for now, probably test best parameter through cross-validation
num_samples = len(train_lab)
embedding_size = 100 # also just for now..
num_epochs = 100
num_batch = 32 # also find optimal through cross-validation

# MAKE WORD ID DICTS
word_to_idx = {}
for i in train+dev:
    # print(i)
    sent = word_tokenize(i.lower())
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)+1 # making the first id is 1, so that I can pad with zeroes.
# Do I need an unknown token when just doing word-to-index (without any counts and such)?
# maybe use keras tokenizer and texts_to_sequences function, which skips unknown words.
# however, an unknown token may be prefered?
vocab_size = len(word_to_idx)
print(vocab_size)
idx_to_word = {v: k for k, v in word_to_idx.items()}

# save Dict file containing the mapping from word ID to word (e.g. train.dict)
f = open("dict.txt","w+")
f.write( str(idx_to_word) )
f.close()
# use tool from lstmvis to transform txt to .Dict file.

# Reshape y_train:
y_train_tiled = np.tile(train_lab, (num_time_steps,1))
y_train_tiled = y_train_tiled.reshape(len(train_lab), num_time_steps , 1)

num_samples = len(train_lab)
# num_cells: number of LSTM cells
num_cells = 100 # 100 for now, probably test best parameter through cross-validation
embedding_size = 100 # also just for now..
num_epochs = 20
num_batch = 32 # also find optimal through cross-validation
print("Parameters:: num_cells: "+str(num_cells)+" num_samples: "+str(num_samples)+" embedding_size: "+str(embedding_size)+" epochs: "+str(num_epochs)+" batch_size: "+str(num_batch))


# max_doc_length vectors of size embedding_size
myInput = Input(shape=(max_doc_length,), name='input')
x = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=max_doc_length)(myInput)
lstm_out = LSTM(num_cells, return_sequences=True)(x)
predictions = TimeDistributed(Dense(2, activation='softmax'))(lstm_out)
model = Model(inputs=myInput, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit({'input': seq}, y_train_tiled, epochs=num_epochs, batch_size=num_batch, verbose=1)

model.layers.pop();
model.summary()
# Save the states via predict
inp = model.input
out = model.layers[-1].output
model_RetreiveStates = Model(inp, out)
states_model = model_RetreiveStates.predict(trainTextsSeq, batch_size=num_batch)
print(states_model.shape)

# Flatten first and second dimension for LSTMVis
states_model_flatten = states_model.reshape(num_samples * num_time_steps, num_cells)

hf = h5py.File("states.hdf5", "w")
hf.create_dataset('states1', data=states_model_flatten)
hf.close()
