import numpy as np
import codecs
#import glob
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import word_tokenize

# datafile

# vocab_size: number of tokens in vocabulary
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
# num_cells: number of LSTM cells
# num_samples: number of training/testing data samples
# num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
# trainTextsSeq: List of input sequence for each document (A matrix with size num_samples * max_doc_length)
# y_train: vector of document class labels

data = codecs.open("thesis/data/kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
data = data[:20800]
data = [s.lower() for s in data]

labels = codecs.open("thesis/data/kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
labels = labels[:20800]
labels = [int(i) for i in labels]

tr, te, trlab, telab = train_test_split(data, labels, test_size=0.33, random_state=42)

word_to_idx = {}
for i in tr+te:
    # print(i)
    sent = word_tokenize(i.lower())
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

#print(word_to_idx)

vocab_size = len(word_to_idx)
