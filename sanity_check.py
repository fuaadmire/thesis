import numpy as np
import codecs
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import nltk
import matplotlib.pyplot as plt
#from keras.utils import plot_model
import datetime
from write_dict_file import d_write
from preprocess_text import preprocess
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
