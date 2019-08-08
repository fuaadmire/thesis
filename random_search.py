from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed, Dropout, CuDNNLSTM, Bidirectional
from keras.models import Model, load_model, Sequential
from thesis.preprocess_text import preprocess
from keras.utils import plot_model, multi_gpu_model # for data parallelism
from keras.constraints import NonNeg
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import codecs

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split

from tensorflow import set_random_seed

import pandas as pd
import h5py
import nltk
nltk.download('punkt')
#import matplotlib.pyplot as plt
import datetime

import gensim
#from thesis.get_liar_binary_data import *
import random
import sys

from write_dict_file import d_write
from my_utils import binarize_labels, load_liar_data, load_kaggle_data, tile_reshape, label_switch, remove_duplicates



datapath = "data/"
#directory_path = "/gdrive/My Drive/Thesis/"

TIMEDISTRIBUTED = False

use_pretrained_embeddings = True

FAKE=1

trainingdata = sys.argv[1] # "liar" or "kaggle"
print("TRAINING WITH", trainingdata)

NUM_LAYERS = sys.argv[2]
print("NUMBER OF LAYERS:", NUM_LAYERS)

# kør med forskellige random seeds og tag gennemsnit. eller cross-validate.
random.seed(16)
np.random.seed(16)
set_random_seed(16)



if trainingdata == "liar":
    train, dev, test, train_lab, dev_lab, test_lab = load_liar_data(datapath)
elif trainingdata == "kaggle":
    train, test, train_lab, test_lab = load_kaggle_data(datapath)
elif trainingdata == "FNC":
    train, test, train_lab, test_lab = load_FNC_data(datapath)

train = [nltk.word_tokenize(i.lower()) for i in train]

test = [nltk.word_tokenize(i.lower()) for i in test]

if trainingdata == "liar":
    dev = [nltk.word_tokenize(i.lower()) for i in dev]


all_train_tokens = []
for i in train:
    for word in i:
        all_train_tokens.append(word)

vocab = set(all_train_tokens)
word2id = {word: i+1 for i, word in enumerate(vocab)}# making the first id is 1, so that I can pad with zeroes.
word2id["UNK"] = len(word2id)+1
id2word = {v: k for k, v in word2id.items()}


#trainTextsSeq: List of input sequence for each document (A matrix with size num_samples * max_doc_length)
trainTextsSeq = np.array([[word2id[w] for w in sent] for sent in train])

testTextsSeq = np.array([[word2id.get(w, word2id["UNK"]) for w in sent] for sent in test])

if trainingdata == "liar":
    devTextsSeq = np.array([[word2id.get(w, word2id["UNK"]) for w in sent] for sent in dev])

# PARAMETERS
vocab_size = len(word2id)+1 # vocab_size: number of tokens in vocabulary
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
max_doc_length = 100 # LIAR 100 (like Wang), Kaggle 3391, FakeNewsCorpus 2669
num_samples = len(train_lab) # num_samples: number of training/testing data samples
num_time_steps = max_doc_length # num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
embedding_size = 300
num_cells = 64 # num_cells: number of LSTM cells
num_epochs = 10
num_batch = 64

# PREPARING DATA

# padding with max doc lentgh
seq = sequence.pad_sequences(trainTextsSeq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)
print("train seq shape",seq.shape)
test_seq = sequence.pad_sequences(testTextsSeq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)
if trainingdata == "liar":
    dev_seq = sequence.pad_sequences(devTextsSeq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)

if TIMEDISTRIBUTED:
    train_lab = tile_reshape(train_lab, num_time_steps)
    test_lab = tile_reshape(test_lab, num_time_steps)
    print(train_lab.shape)
    if trainingdata == "liar":
        dev_lab = tile_reshape(dev_lab, num_time_steps)
else:
    train_lab = to_categorical(train_lab, 2)
    test_lab = to_categorical(test_lab, 2)
    print(train_lab.shape)
    if trainingdata == "liar":
        dev_lab = to_categorical(dev_lab, 2)

#print("Parameters:: num_cells: "+str(num_cells)+" num_samples: "+str(num_samples)+" embedding_size: "+str(embedding_size)+" epochs: "+str(num_epochs)+" batch_size: "+str(num_batch))


if use_pretrained_embeddings:
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    embedding_matrix = np.zeros((len(word2id) + 1, 300))
    for word, i in word2id.items():
        try:
            embedding_vector = model.wv[word]
        except:
            embedding_vector = model.wv["UNK"]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# Create model for KerasClassifier
def create_model(num_cells,
                 dropout,
                 r_dropout,
                 learning_rate,
                 #input_dim=vocab_size,
                 #output_dim=embedding_size,
                 #input_length=max_doc_length
                 ):
    # Model definition
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix],input_length=max_doc_length,trainable=True))
    #model.add(LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True, kernel_constraint=NonNeg()))
    if NUM_LAYERS==1:
        model.add(Bidirectional(LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout, kernel_constraint=NonNeg())))
    elif NUM_LAYERS==2: # stacked LSTM
        model.add(Bidirectional(LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout,return_sequences=True, kernel_constraint=NonNeg())))
        model.add(LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout, kernel_constraint=NonNeg()))
    else:
        print("number of layers not specified properly")
    #model.add(TimeDistributed(Dense(1, activation='sigmoid', kernel_constraint=NonNeg())))
    #model.add(Dense(1, activation='sigmoid', kernel_constraint=NonNeg()))
    model.add(Dense(2, activation='softmax', kernel_constraint=NonNeg()))
    opt = Adam(lr=learning_rate)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer=opt,
                  #loss='binary_crossentropy',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return parallel_model

model = KerasClassifier(build_fn=create_model)

# Specify parameters and distributions to sample from
num_cells = [32,64,128,256]
#hparams2 = ['elu', 'relu', ...]
dropout = [0.2,0.4,0.6,0.8]
r_dropout = [0.2,0.4,0.6,0.8]
learning_rate = [0.01, 0.001, 0.0001]
epochs = [10,30]

# Prepare the Dict for the Search
param_dist = dict(num_cells=num_cells,
                  dropout=dropout,
                  r_dropout=r_dropout,
                  #batch_size=batch_size,
                  learning_rate=learning_rate,
                  epochs=epochs,
                  #verbose=[1]
                 )

if trainingdata == "liar":
    my_val_fold = [-1 for i in range(len(seq))]+[0 for i in range(len(dev_seq))]

    X = np.concatenate((seq, dev_seq))
    y = np.concatenate((train_lab, dev_lab))

    ps = PredefinedSplit(test_fold=my_val_fold)
else:
    X = seq
    y = train_lab
    ps=2

# Search in action!
n_iter_search = 96 # Number of parameter settings that are sampled.
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   #n_jobs=1,
								   cv=ps,
								   verbose=2)

random_search.fit(X, y)

# Show the results
print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
params = random_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
