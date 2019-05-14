from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed, Dropout
from keras.models import Model
from preprocess_text import preprocess
from keras.utils import multi_gpu_model # for data parallelism
from keras.constraints import NonNeg
from keras import regularizers
from keras.utils import plot_model
import numpy as np
import codecs
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import nltk
#import matplotlib.pyplot as plt
import datetime
from write_dict_file import d_write
#import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = "device=cuda"
#os.environ['floatX']='float32'

print(datetime.datetime.now())

data = codecs.open("data/kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
data = data[:20800]
#data = data[:200]
data = [s.lower() for s in data]
labels = codecs.open("data/kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
labels = labels[:20800]
#labels = labels[:200]
labels = [int(i) for i in labels]

# disregarding input which is less than 100 characters (as they do not contain many words, if any)
labels_include = []
data_include = []
for indel, i in enumerate(data):
    if len(i) > 100:
        data_include.append(i)
        labels_include.append(labels[indel])


train, dev, train_lab, dev_lab = train_test_split(data_include, labels_include, test_size=0.33, random_state=42)

#preprocessing is going wrong. returns letters instead of words :)
#train = preprocess(train)
#dev = preprocess(dev)

train = [nltk.word_tokenize(i.lower()) for i in train]
dev = [nltk.word_tokenize(i.lower()) for i in dev]

# perhaps edit this to make dict straight away.

all_train_tokens = []
for i in train:
    for word in i:
        all_train_tokens.append(word)

vocab = set(all_train_tokens)
word2id = {word: i+1 for i, word in enumerate(vocab)}# making the first id is 1, so that I can pad with zeroes.
word2id["UNK"] = len(word2id)+1
id2word = {v: k for k, v in word2id.items()}
#[[id2word.get(word, "UNK") for word in sent] for sent in dev]
# save Dict file containing the mapping from word ID to word (e.g. train.dict)
d_write("words.dict", word2id)
#f = open("dict.txt","w+")
#f.write( str(id2word) )
#f.close()


#trainTextsSeq: List of input sequence for each document (A matrix with size num_samples * max_doc_length)
trainTextsSeq = np.array([[word2id[w] for w in sent] for sent in train])
testTextsSeq = np.array([[word2id.get(w, word2id["UNK"]) for w in sent] for sent in dev])

# PARAMETERS
# vocab_size: number of tokens in vocabulary
vocab_size = len(word2id)+1
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
max_doc_length = 1000
# num_cells: number of LSTM cells
num_cells = 64 # for now, probably test best parameter through cross-validation

# num_samples: number of training/testing data samples
num_samples = len(train_lab)
# num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
num_time_steps = max_doc_length
embedding_size = 300 # also just for now..
num_epochs = 10
num_batch = 64 # also find optimal through cross-validation


# PREPARING DATA

# padding with max doc lentgh
seq = sequence.pad_sequences(trainTextsSeq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)
print("train seq shape",seq.shape)
test_seq = sequence.pad_sequences(testTextsSeq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)


trainTextsSeq_flatten = np.array(seq).flatten()
hf = h5py.File("train.hdf5", "w") # need this file for LSTMVis
hf.create_dataset('words', data=trainTextsSeq_flatten)
hf.close()


# Reshape y_train:
def tile_reshape(train_lab, num_time_steps):
    y_train_tiled = np.tile(train_lab, (num_time_steps,1)).T
    y_train_tiled = y_train_tiled.reshape(len(train_lab), num_time_steps , 1)
    #print("y_train_shape:",y_train_tiled.shape)
    return y_train_tiled

y_train_tiled = tile_reshape(train_lab, num_time_steps)
y_test_tiled = tile_reshape(dev_lab, num_time_steps)
print("Parameters:: num_cells: "+str(num_cells)+" num_samples: "+str(num_samples)+" embedding_size: "+str(embedding_size)+" epochs: "+str(num_epochs)+" batch_size: "+str(num_batch))
#print(y_train_tiled)


myInput = Input(shape=(max_doc_length,), name='input')
print(myInput.shape)
x = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_doc_length)(myInput)
print(x.shape)
lstm_out = LSTM(num_cells, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, kernel_constraint=NonNeg())(x)
print(lstm_out.shape)
#out = TimeDistributed(Dense(2, activation='softmax'))(lstm_out)
predictions = TimeDistributed(Dense(1, activation='sigmoid', kernel_constraint=NonNeg()))(lstm_out) #kernel_constraint=NonNeg()

print("predictions_shape:",predictions.shape)
model = Model(inputs=myInput, outputs=predictions)

# try-except to switch between gpu and cpu version
try:
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("fitting model..")
    parallel_model.fit({'input': seq}, y_train_tiled, epochs=num_epochs, verbose=2, batch_size=num_batch, validation_split=.20)
    #parallel_model.fit(seq, y_train_tiled, epochs=num_epochs, verbose=2, steps_per_epoch=(np.int(np.floor(num_samples/num_batch))), validation_split=.20) # or try this, removing the curly brackets.
    #parallel_model.fit({'input': seq}, train_lab, epochs=num_epochs, batch_size=num_batch, verbose=1)

    print("Testing...")
    score = parallel_model.evaluate(test_seq, y_test_tiled, batch_size=num_batch, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # get predicted classes
    y_prob = parallel_model.predict(seq)
except:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("fitting model..")
    model.fit({'input': seq}, y_train_tiled, epochs=num_epochs, verbose=2, batch_size=num_batch, validation_split=.20)
    #parallel_model.fit(seq, y_train_tiled, epochs=num_epochs, verbose=2, steps_per_epoch=(np.int(np.floor(num_samples/num_batch))), validation_split=.20) # or try this, removing the curly brackets.
    #parallel_model.fit({'input': seq}, train_lab, epochs=num_epochs, batch_size=num_batch, verbose=1)

    print("Testing...")
    score = model.evaluate(test_seq, y_test_tiled, batch_size=num_batch, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # get predicted classes
    y_prob = model.predict(seq)

model.summary()


# if multiclass:
#y_classes = y_prob.argmax(axis=-1)
# if binary:
y_classes = (y_prob > 0.5).astype(np.int)
print(y_classes)
print(y_classes.shape)
print(type(y_classes))
# save predicted classes!
predicted_classes_flatten = np.array(y_classes).flatten()
hf_p = h5py.File("predictions.hdf5", "w")
hf_p.create_dataset('preds', data=predicted_classes_flatten)
hf_p.close()



try:
    model.layers.pop();
    inp = model.inputs
    out = model.layers[-1].output
    model_RetreiveStates = Model(inp, out)

    # get predicted classes
    train_preds = model.predict(seq)
    test_preds = model.predict(test_seq)

    # Save plot of model
    plot_model(model, to_file="model.png")

    #first_unit_preds = states_result[:,:,0]
    #second_unit_preds = states_result[:,:,1]

    #first_unit = np.array(first_unit_preds).flatten()
    #hf_states1_p = h5py.File("states_pred_1.hdf5", "w")
    #hf_states1_p.create_dataset('preds', data=first_unit)
    #hf_states1_p.close()

    #second_unit = np.array(second_unit_preds).flatten()
    #hf_states2_p = h5py.File("states_pred_2.hdf5", "w")
    #hf_states2_p.create_dataset('preds', data=second_unit)
    #hf_states2_p.close()

except:
    parallel_model.layers.pop();
    inp = parallel_model.inputs
    out = parallel_model.layers[-1].output
    model_RetreiveStates = Model(inp, out)

    # get predicted classes
    train_preds = parallel_model.predict(seq)
    test_preds = parallel_model.predict(test_seq)

    plot_model(parallel_model, to_file="model.png")
    #first_unit_preds = states_result[:,:,0]
    #second_unit_preds = states_result[:,:,1]

    #first_unit = np.array(first_unit_preds).flatten()
    #hf_states1_p = h5py.File("states_pred_1.hdf5", "w")
    #hf_states1_p.create_dataset('preds', data=first_unit)
    #hf_states1_p.close()

    #second_unit = np.array(second_unit_preds).flatten()
    #hf_states2_p = h5py.File("states_pred_2.hdf5", "w")
    #hf_states2_p.create_dataset('preds', data=second_unit)
    #hf_states2_p.close()

# Getting all the files

# text files
trainTextsSeq_flatten = np.array(seq).flatten()
hf = h5py.File("train.hdf5", "w") # need this file for LSTMVis
hf.create_dataset('words', data=trainTextsSeq_flatten)
hf.close()

testTextsSeq_flatten = np.array(test_seq).flatten()
hf_t = h5py.File("test.hdf5", "w") # need this file for LSTMVis
hf_t.create_dataset('words', data=testTextsSeq_flatten)
hf_t.close()



# predictions

# if multiclass:
#y_classes_train = train_preds.argmax(axis=-1)
#y_classes_test = test_preds.argmax(axis=-1)
# if binary:
#y_classes = (y_prob > 0.5).astype(np.int)
y_classes_train = (train_preds > 0.5).astype(np.int)
y_classes_test = (test_preds > 0.5).astype(np.int)

# save predicted classes!
predicted_train_classes_flatten = np.array(y_classes_train).flatten()
hf_p = h5py.File("predictions.hdf5", "w")
hf_p.create_dataset('preds_train', data=predicted_train_classes_flatten)
#hf_p.close()
predicted_test_classes_flatten = np.array(y_classes_test).flatten()
hf_p.create_dataset('preds_test', data=predicted_test_classes_flatten)
#hf_pt = h5py.File("test_predictions.hdf5", "w")
#hf_pt.create_dataset('preds', data=predicted_test_classes_flatten)
test_true_classes = np.array(y_test_tiled).flatten()
hf_p.create_dataset('true_classes_testset', data=test_true_classes)
train_true_classes = np.array(y_train_tiled).flatten()
hf_p.create_dataset('true_classes_trainset', data=train_true_classes)
hf_p.close()

# dictionary
d_write("words.dict", word2id)
d_write("words.dict", {"PADDING": 0}) # add padding token to lstmvis dict


states_model = model_RetreiveStates.predict(seq, batch_size=num_batch)
states_model_test = model_RetreiveStates.predict(test_seq, batch_size=num_batch)
states_model_flatten = states_model.reshape(num_samples * num_time_steps, num_cells)# Flatten first and second dimension for LSTMVis
hf = h5py.File("states.hdf5", "w")
hf.create_dataset('states_train', data=states_model_flatten)
states_model_test_flatten = states_model_test.reshape(len(test_seq) * num_time_steps, num_cells)# Flatten first and second dimension for LSTMVis
hf.create_dataset('states_test', data=states_model_test_flatten)
hf.close()


# Save plot of model
#plot_model(model, to_file="model.png")
