import pandas as pd
import tqdm
import numpy as np
import codecs
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import random
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

random.seed(16)
np.random.seed(16)

def binarize_labels(labels):
    labels_transformed = [1 if i in [2,3,5] else 0 for i in labels]
    return labels_transformed

def load_liar_data():

    liar_train = codecs.open("data/liar_xtrain.txt", 'r', 'utf-8').read().split('\n')
    liar_train = [s.lower() for s in liar_train if len(s) > 1]
    liar_train_labels = codecs.open('data/liar_ytrain.txt', 'r', 'utf-8').read().split('\n')
    liar_train_lab = [s for s in liar_train_labels if len(s) > 1]

    liar_dev = codecs.open("data/liar_xval.txt", 'r', 'utf-8').read().split('\n')
    liar_dev = [s.lower() for s in liar_dev if len(s) > 1]
    liar_dev_labels = codecs.open("data/liar_yval.txt", 'r', 'utf-8').read().split('\n')
    liar_dev_lab = [s for s in liar_dev_labels if len(s) > 1]

    liar_test = codecs.open("data/liar_xtest.txt", 'r', 'utf-8').read().split('\n')
    liar_test = [s.lower() for s in liar_test if len(s) > 1]
    liar_test_labels = codecs.open("data/liar_ytest.txt", 'r', 'utf-8').read().split('\n')
    liar_test_lab = [s for s in liar_test_labels if len(s) > 1]

    assert len(liar_train) == len(liar_train_lab)
    assert len(liar_dev) == len(liar_dev_lab)
    assert len(liar_test) == len(liar_test_lab)

    le = preprocessing.LabelEncoder()
    #classes = ['pants-fire','false','barely-true','half-true','mostly-true','true']
    #le.fit_transform(classes)
    liar_train_lab = le.fit_transform(liar_train_lab)
    liar_dev_lab = le.transform(liar_dev_lab)
    liar_test_lab = le.transform(liar_test_lab)

    print(le.classes_) #['barely-true' 'false' 'half-true' 'mostly-true' 'pants-fire' 'true']
    print(le.transform(le.classes_)) # [0 1 2 3 4 5]
    # untrue classes (to be encoded as 0): 4, 1, 0
    # true classes (to be encoded as 1): 2, 3, 5

    liar_train_lab = binarize_labels(liar_train_lab)
    liar_dev_lab = binarize_labels(liar_dev_lab)
    liar_test_lab = binarize_labels(liar_test_lab)

    return liar_train, liar_dev, liar_test, liar_train_lab, liar_dev_lab, liar_test_lab
