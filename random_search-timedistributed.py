from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed, Dropout, CuDNNLSTM, Bidirectional, concatenate
from keras.models import Model, load_model, Sequential
#from thesis.preprocess_text import preprocess
from keras.utils import multi_gpu_model, plot_model
from keras.constraints import NonNeg
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras import callbacks

import numpy as np
import codecs
import sys
sys.path.append('thesis/')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.utils import shuffle

import pandas as pd
import h5py
import nltk
nltk.download('punkt')
#import matplotlib.pyplot as plt
import datetime
from thesis.write_dict_file import d_write
import gensim
import random

from my_data_utils import load_liar_data, tile_reshape, load_kaggle_data, load_FNC_data, load_BS_data, load_pos_tags

import matplotlib.pyplot as plt

from my_model_utils import pre_modelling_stuff, evaluting_model

random.seed(42)
np.random.seed(42)

from tensorflow import set_random_seed
set_random_seed(42)

trainingdata = sys.argv[1]

TIMEDISTRIBUTED=True


embedding_matrix, seq, test_seq, dev_seq, train_lab, test_lab, dev_lab, vocab_size = pre_modelling_stuff(TIMEDISTRIBUTED=TIMEDISTRIBUTED,
                                                                                                        trainingdata=trainingdata)
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
max_doc_length = 100 # LIAR 100 (like Wang), Kaggle 3391, FakeNewsCorpus 2669
num_samples = len(train_lab) # num_samples: number of training/testing data samples
num_time_steps = max_doc_length # num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
embedding_size = 300
num_batch = 64

def create_model(
                 num_cells,
                 dropout,
                 r_dropout,
                 learning_rate,
                 num_epochs,
                 seq,
                 train_lab,
                 dev_seq,
                 dev_lab,
                 input_dim,
                 output_dim,
                 input_length
                 ):

    K.clear_session()

    myInput = Input(shape=(max_doc_length,), name='input')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix],input_length=max_doc_length,trainable=True)(myInput)
    lstm_out = LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True, kernel_constraint=NonNeg())(x)
    predictions = TimeDistributed(Dense(1, activation='sigmoid', kernel_constraint=NonNeg()))(lstm_out)
    model = Model(inputs=myInput, outputs=predictions)
    opt = Adam(lr=learning_rate)
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  #loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #if trainingdata == "liar":
    model.fit({'input': seq}, train_lab, epochs=num_epochs, verbose=2, batch_size=num_batch, validation_data=(dev_seq,dev_lab))
    #else:
    #    model.fit({'input': seq}, train_lab, epochs=num_epochs, verbose=2, batch_size=num_batch)
    #print("Testing...")
    #test_score = model.evaluate(test_seq, test_lab, batch_size=num_batch, verbose=0)
    #if trainingdata == "liar":
    dev_score = model.evaluate(dev_seq, dev_lab, batch_size=num_batch, verbose=0)
    #print("Test loss:", test_score[0])
    #print("Test accuracy:", test_score[1])
    return dev_score[1]



# Specify parameters and distributions to sample from
num_cells_list = [32,64,128,256] #first
#num_cells = [32] # second
dropout_list = [0.2,0.4,0.6] #first
#dropout = [0.4] # second
r_dropout_list = [0.2,0.4,0.6] # first
#r_dropout = [0.4] # second
learning_rate_list = [0.001, 0.0001, 0.00001] # first
#learning_rate = [0.0001, 0.00001] #second
epochs_list = [10] # first
#epochs = [10,100] # second


# Search in action!
n_iter_search = 25 # Number of parameter settings that are sampled.
if TIMEDISTRIBUTED:
    params = []
    accuracies = []
    for i in range(n_iter_search):
        num_cells = np.random.choice(num_cells_list)
        dropout = np.random.choice(dropout_list)
        r_dropout = np.random.choice(r_dropout_list)
        learning_rate = np.random.choice(learning_rate_list)
        epochs = np.random.choice(epochs_list)
        params.append(["num_cells:", num_cells, "dropout:", dropout, "r_dropout:", r_dropout, "learning_rate:", learning_rate])

        print("num_cells:", num_cells, "dropout:", dropout, "r_dropout:", r_dropout, "learning_rate:", learning_rate)
        accuracy = create_model(num_cells, dropout, r_dropout, learning_rate, epochs, seq, train_lab, dev_seq, dev_lab, vocab_size, embedding_size,max_doc_length)
        print(accuracy)
        accuracies.append(accuracy)

for i,j in zip(accuracies, params):
    print(i,j)
