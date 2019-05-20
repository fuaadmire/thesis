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


# FakeNewsCorpus data (part of it, 25000 samples in each class)
FNC_fake = codecs.open("data/fake_part1.txt", 'r', 'utf-8').read().split('\n')
FNC_fake = FNC_fake[:25000]
FNC_true = codecs.open("data/true_part1.txt", 'r', 'utf-8').read().split('\n')
FNC_true = FNC_fake[:25000]
FNC_fake_labels = np.zeros(len(FNC_fake))
FNC_true_labels = np.ones(len(FNC_true))
FNC_samples = np.concatenate((FNC_fake, FNC_true))
FNC_labels = np.concatenate((FNC_fake_labels, FNC_true_labels))
assert len(FNC_samples) == len(FNC_labels)
FNC_samples, FNC_labels = shuffle(FNC_samples, FNC_labels, random_state=42)

FNC_Xtrain, FNC_Xtest, FNC_ytrain, FNC_ytest = train_test_split(FNC_samples, FNC_labels, test_size=0.33, random_state=42)

with open("data/FakeNewsCorpus_Xtest.txt", "w+") as savefile:
    for sample in FNC_Xtest:
        savefile.write(sample)
        savefile.write("\n")
#np.savetxt("data/FakeNewsCorpus_Xtest", FNC_Xtest)
np.savetxt("data/FakeNewsCorpus_ytest.txt", FNC_ytest)
