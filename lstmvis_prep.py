import numpy as np
import codecs
#import glob
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import word_tokenize
import h5py
import os
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed
from keras.models import Model
from preprocess_text import preprocess


# DATA

data = codecs.open("data/kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
data = data[:20800]
data = [s.lower() for s in data]

labels = codecs.open("data/kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
labels = labels[:20800]
labels = np.array([int(i) for i in labels])

train, dev, train_lab, dev_lab = train_test_split(data, labels, test_size=0.33, random_state=42)



# instead do preprocessing but do it once and save files
train = preprocess(train)
dev = preprocess(dev)

#train = [word_tokenize(i.lower()) for i in train]
#dev = [word_tokenize(i.lower()) for i in dev]




# MAKE VOCAB AND WORD ID DICTS

all_train_tokens = []
for i in train:
    for word in i:
        all_train_tokens.append(word)

vocab = set(all_train_tokens)
word2id = {word: i+1 for i, word in enumerate(vocab)}# making the first id is 1, so that I can pad with zeroes.
word2id["UNK"] = len(word2id)+1
id2word = {v: k for k, v in word2id.items()}
[[id2word.get(word, "UNK") for word in sent] for sent in dev]

# save Dict file containing the mapping from word ID to word (e.g. train.dict)
f = open("dict.txt","w+")
f.write( str(id2word) )
f.close()
# use tool from lstmvis to transform txt to .Dict file.





# PARAMETERS

# vocab_size: number of tokens in vocabulary
vocab_size = len(vocab)+1 # +1 for oov /  unknown token
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
max_doc_length = int(np.round(np.mean([len(paragraph) for paragraph in train]))) # using the mean length of documents as max_doc_length for now
# num_cells: number of LSTM cells
num_cells = 32 # for now, probably test best parameter through cross-validation
# num_samples: number of training/testing data samples
num_samples = len(train_lab)
# num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
num_time_steps = max_doc_length

embedding_size = 20 # also just for now..
num_epochs = 10
num_batch = 16 # also find optimal through cross-validation


# PREPARING TRAIN DATA

# trainTextsSeq: List of input sequence for each document (A matrix with size num_samples * max_doc_length)
trainTextsSeq_list = []
for input_sequence in train:
    inputs = [word2id[w] for w in input_sequence]
    trainTextsSeq_list.append(inputs)
trainTextsSeq = np.array(trainTextsSeq_list)

# padding with max doc lentgh (mean length at the moment)
seq = sequence.pad_sequences(trainTextsSeq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)
print("train seq shape",seq.shape)



trainTextsSeq_flatten = np.array(seq).flatten()
hf = h5py.File("train.hdf5", "w") # need this file for LSTMVis
hf.create_dataset('words', data=trainTextsSeq_flatten)
hf.close()

# Reshape y_train:
y_train_tiled = np.tile(train_lab, (num_time_steps,1))
y_train_tiled = y_train_tiled.reshape(len(train_lab), num_time_steps , 1)
print("y_train_shape:",y_train_tiled.shape)


print("Parameters:: num_cells: "+str(num_cells)+" num_samples: "+str(num_samples)+" embedding_size: "+str(embedding_size)+" epochs: "+str(num_epochs)+" batch_size: "+str(num_batch))

#seq=seq.reshape(seq.shape[0],seq.shape[1],1)
# max_doc_length vectors of size embedding_size
myInput = Input(shape=(max_doc_length,), name='input')
print(myInput.shape)
x = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_doc_length)(myInput)
print(x.shape)
lstm_out = LSTM(num_cells, return_sequences=True)(x)
print(lstm_out.shape)
predictions = TimeDistributed(Dense(2, activation='softmax'))(lstm_out)
print("predictions_shape:",predictions.shape)
model = Model(inputs=myInput, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit({'input': seq}, y_train_tiled, epochs=num_epochs, verbose=2, steps_per_epoch=(np.int(np.floor(num_samples/num_batch))))
#model.fit({'input': seq}, train_lab, epochs=num_epochs, batch_size=num_batch, verbose=1)

model.layers.pop();
model.summary()
# Save the states via predict
inp = model.input
out = model.layers[-1].output
model_RetreiveStates = Model(inp, out)
states_model = model_RetreiveStates.predict(seq, batch_size=num_batch)
print(states_model.shape)

# Flatten first and second dimension for LSTMVis
states_model_flatten = states_model.reshape(num_samples * num_time_steps, num_cells)

hf = h5py.File("states.hdf5", "w")
hf.create_dataset('states1', data=states_model_flatten)
hf.close()
