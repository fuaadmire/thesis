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

m= 5000 #number of feats 5000 or 10000
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

def print_scores(y, y_hat, string):
    print(string)
    print("binary F1", f1_score(y, y_hat))
    print("micro f1:",f1_score(y, y_hat, average='micro'))
    print("macro f1:", f1_score(y, y_hat, average='macro'))
    print("weighted F1", f1_score(y, y_hat, average='weighted'))
    print("accuracy", accuracy_score(y, y_hat))
    print()

liar_train_lab = binarize_label(liar_train_lab)
liar_dev_lab = binarize_label(liar_dev_lab)
liar_test_lab = binarize_label(liar_test_lab)



X_1, X_2, y_1, y_2 = train_test_split(liar_train, liar_train_lab, test_size=0.50, random_state=42)

print(len(y_1))
print(len(y_2))

split1_vect = TfidfVectorizer(ngram_range=(v,k), max_features=m)
X_train1 = split1_vect.fit_transform(X_1)
X_test1 = split1_vect.transform(liar_test)
feats_1 = ['_'.join(s.split()) for s in split1_vect.get_feature_names()]

#y_rand = y_1.copy()
#np.random.shuffle(y_rand)

clf_1 = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(X_train1, y_1)
#clf_1 = LogisticRegression(random_state=16, solver='saga', penalty=None, max_iter=10000).fit(X_train1, y_rand)
coefs_1 = clf_1.coef_
coefs_1_df = pd.DataFrame.from_records(coefs_1, columns=feats_1)
coefs_1_df.to_csv('NEW_liar_train_split1_coefs.csv', sep='\t', index=False)
#coefs_1_df.to_csv('testing_random_labels_coefs_on_larger_data.csv', sep='\t', index=False)


split2_vect = TfidfVectorizer(ngram_range=(v,k), max_features=m)
X_train2 = split2_vect.fit_transform(X_2)
X_test2 = split2_vect.transform(liar_test)
feats_2 = ['_'.join(s.split()) for s in split2_vect.get_feature_names()]
clf_2 = LogisticRegression(random_state=16, solver='saga', penalty=None, max_iter=10000).fit(X_train2, y_2)
coefs_2 = clf_2.coef_
coefs_2_df = pd.DataFrame.from_records(coefs_2, columns=feats_2)
coefs_2_df.to_csv('NEW_liar_train_split2_coefs.csv', sep='\t', index=False)

def print_scores(y, y_hat, string):
    print(string)
    print("binary F1", f1_score(y, y_hat))
    print("micro f1:",f1_score(y, y_hat, average='micro'))
    print("macro f1:", f1_score(y, y_hat, average='macro'))
    print("weighted F1", f1_score(y, y_hat, average='weighted'))
    print("accuracy", accuracy_score(y, y_hat))
    print()

split_1_preds = clf_1.predict(X_test1)
print_scores(liar_test_lab, split_1_preds,"Split 1 Test scores")

split_2_preds = clf_2.predict(X_test2)
print_scores(liar_test_lab, split_2_preds,"Split 2 Test scores")



X = np.concatenate((liar_dev, liar_test))
y = np.concatenate((liar_dev_lab, liar_test_lab))

y_rand = y.copy()
np.random.shuffle(y_rand)

liar_test_vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
liar_rand_test = liar_test_vectorizer.fit_transform(X)
rand_feats = ['_'.join(s.split()) for s in liar_test_vectorizer.get_feature_names()]
clf_rand = None
clf_rand = LogisticRegression(random_state=16, solver='saga', penalty=None, max_iter=10000).fit(liar_rand_test,y_rand)
rand_coefs = clf_rand.coef_
liar_rand_coefs = pd.DataFrame.from_records(rand_coefs, columns=rand_feats)
liar_rand_coefs.to_csv('NEW_liar_testanddev_random_labels.csv', sep='\t', index=False)


print("Done")
