from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed, Dropout, CuDNNLSTM, Bidirectional
from keras.models import Model, load_model
#from thesis.preprocess_text import preprocess
from keras.utils import multi_gpu_model, plot_model
from keras.constraints import NonNeg
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

import numpy as np
import codecs
import sys

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pandas as pd
import h5py
import nltk
#nltk.download('punkt')
#import matplotlib.pyplot as plt
import datetime
from write_dict_file import d_write
import gensim
import random

from my_data_utils import load_liar_data, tile_reshape, load_kaggle_data, load_FNC_data, load_BS_data

import matplotlib.pyplot as plt

def plot_loss(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('loss200lr0001.png', dpi=300)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("accuracy200lr0001.png", dpi=300)


def train_and_test(TIMEDISTRIBUTED=False,
                    trainingdata="liar",
                    num_cells=32,
                    num_epochs=10,
                    dropout=0.4,
                    r_dropout=0.4,
                    num_batch=64,
                    learning_rate=0.0001):

    datapath = "/home/ktj250/thesis/data/"
    #directory_path = "/gdrive/My Drive/Thesis/"

    #TIMEDISTRIBUTED = False

    use_pretrained_embeddings = True

    FAKE=1

    #trainingdata = sys.argv[1] #"liar" # kaggle, FNC, BS

    print("trainingdata=",trainingdata)

    if trainingdata == "liar":
        train, dev, test, train_lab, dev_lab, test_lab = load_liar_data(datapath)
    elif trainingdata == "kaggle":
        train, test, train_lab, test_lab = load_kaggle_data(datapath)
    elif trainingdata == "FNC":
        train, test, train_lab, test_lab = load_FNC_data(datapath)
    elif trainingdata == "BS":
        train, test, train_lab, test_lab = load_BS_data(datapath)

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
    # vocab_size: number of tokens in vocabulary
    vocab_size = len(word2id)+1
    # max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
    max_doc_length = 100 # LIAR 100 (like Wang), Kaggle 3391, FakeNewsCorpus 2669
    # num_samples: number of training/testing data samples
    num_samples = len(train_lab)
    # num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
    num_time_steps = max_doc_length
    embedding_size = 300 # also just for now..

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

    print("Parameters:: num_cells: "+str(num_cells)+" num_samples: "+str(num_samples)+" embedding_size: "+str(embedding_size)+" epochs: "+str(num_epochs)+" batch_size: "+str(num_batch))


    if use_pretrained_embeddings:
        # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        # Load Google's pre-trained Word2Vec model.
        model = gensim.models.KeyedVectors.load_word2vec_format('/home/ktj250/thesis/GoogleNews-vectors-negative300.bin', binary=True)

        embedding_matrix = np.zeros((len(word2id) + 1, 300))
        for word, i in word2id.items():
            try:
                embedding_vector = model.wv[word]
            except:
                embedding_vector = model.wv["UNK"]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    myInput = Input(shape=(max_doc_length,), name='input')
    print(myInput.shape)
    if use_pretrained_embeddings:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix],input_length=max_doc_length,trainable=True)(myInput)
    else:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_doc_length)(myInput)
        print(x.shape)

    if TIMEDISTRIBUTED:
        lstm_out = LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True, kernel_constraint=NonNeg())(x)
        predictions = TimeDistributed(Dense(1, activation='sigmoid', kernel_constraint=NonNeg()))(lstm_out)
    else:
        lstm_out = Bidirectional(LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout))(x)
        predictions = Dense(2, activation='softmax')(lstm_out)

    model = Model(inputs=myInput, outputs=predictions)

    opt = Adam(lr=learning_rate)

    if TIMEDISTRIBUTED:
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("fitting model..")
    if trainingdata == "liar":
        history = model.fit({'input': seq}, train_lab, epochs=num_epochs, verbose=2, batch_size=num_batch, validation_data=(dev_seq,dev_lab))
    else:
        history = model.fit({'input': seq}, train_lab, epochs=num_epochs, verbose=2, batch_size=num_batch)
    print("Testing...")
    test_score = model.evaluate(test_seq, test_lab, batch_size=num_batch, verbose=0)
    if trainingdata == "liar":
        dev_score = model.evaluate(dev_seq, dev_lab, batch_size=num_batch, verbose=0)

    print("Test loss:", test_score[0])
    print("Test accuracy:", test_score[1])
    if trainingdata == "liar":
        print("Valid loss:", dev_score[0])
        print("Valid accuracy:", dev_score[1])

    if not TIMEDISTRIBUTED:
        preds = model.predict(test_seq)
        f1 = f1_score(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1))
        tn, fp, fn, tp = confusion_matrix(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1)).ravel()
        print("tn, fp, fn, tp")
        print(tn, fp, fn, tp)


    model.summary()

    if trainingdata=="liar":
        return dev_score, history
    else:
        return test_score, history
