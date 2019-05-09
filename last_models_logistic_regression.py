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

m= 10000 #number of feats 5000 or 10000
k=5#max ngram
v=1 #min mgram

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

def binarize_label(labels):
    labels_transformed = [1 if i in [2,3,5] else 0 for i in labels]
    return labels_transformed

liar_train_lab = binarize_label(liar_train_lab)
liar_dev_lab = binarize_label(liar_dev_lab)
liar_test_lab = binarize_label(liar_test_lab)

def print_scores(y, y_hat, string):
    print(string)
    print("binary F1", f1_score(y, y_hat))
    print("micro f1:",f1_score(y, y_hat, average='micro'))
    print("macro f1:", f1_score(y, y_hat, average='macro'))
    print("weighted F1", f1_score(y, y_hat, average='weighted'))
    print("accuracy", accuracy_score(y, y_hat))
    print()

# Vectorizing
liar_vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
#X_train_liar = liar_vectorizer.fit_transform(liar_train)
X_dev_liar = liar_vectorizer.fit_transform(liar_dev)
X_test_liar = liar_vectorizer.transform(liar_test)
liar_feats = ['_'.join(s.split()) for s in liar_vectorizer.get_feature_names()]

# Træn logreg og få coefs på Liar valid
clf_liar=None
clf_liar = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(X_dev_liar,liar_dev_lab)
liar_coefs = clf_liar.coef_
allcoefs_liar = pd.DataFrame.from_records(liar_coefs, columns=liar_feats) #add ngrams as colnames
allcoefs_liar.to_csv('liar_valid_coefs_final.csv', sep='\t', index=False)

preds_liar_test = clf_liar.predict(X_test_liar)
print_scores(liar_test_lab, preds_liar_test, "Liar test set prediction scores by classifier trained on liar valid." )

# træn logreg Classifier med liar X_test og random labels
# random liar test set
random_liar_labels = liar_test_lab.copy()
np.random.shuffle(random_liar_labels)

liar_test_vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
liar_rand_test = liar_test_vectorizer.fit_transform(liar_test)
rand_feats = ['_'.join(s.split()) for s in liar_test_vectorizer.get_feature_names()]
clf_rand = None
clf_rand = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(liar_rand_test,random_liar_labels)
rand_coefs = clf_rand.coef_
liar_rand_coefs = pd.DataFrame.from_records(rand_coefs, columns=rand_feats)
liar_rand_coefs.to_csv('liar_test_shuffled_labels_coefs_final.csv', sep='\t', index=False)

print("done")




# brug nyt dataset?
