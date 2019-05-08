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

def label_switch(labels):
    labels_transformed = [1 if i==0 else 0 for i in labels]
    return labels_transformed


liar_train_lab = binarize_label(liar_train_lab)
liar_dev_lab = binarize_label(liar_dev_lab)
liar_test_lab = binarize_label(liar_test_lab)


# Kaggle data
data = codecs.open("data/kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
data = data[:20800]
data = [s.lower() for s in data]

labels = codecs.open("data/kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
labels = labels[:20800]
labels = [int(i) for i in labels]

tr, te, trlab, telab = train_test_split(data, labels, test_size=0.33, random_state=42)

kaggle_train = tr
kaggle_train_lab = label_switch(trlab)
kaggle_test = te
kaggle_test_lab = label_switch(telab)


# FakeNewsNet Politifact data
real = codecs.open("data/RealNewsContent.txt", 'r', 'utf-8').read().split('\n')
real = [s.lower() for s in real if len(s) > 1]
fake = codecs.open("data/FakeNewsContent.txt", 'r', 'utf-8').read().split('\n')
fake = [s.lower() for s in fake if len(s) > 1]
real_labels = np.ones(len(real))
fake_labels = np.zeros(len(fake))
FNN_labels = np.concatenate((real_labels, fake_labels))
FNN_texts = np.concatenate((real, fake))
assert len(FNN_labels) == len(FNN_texts)
FNN_X, FNN_y = shuffle(FNN_texts,FNN_labels, random_state=42)



m= 10000 #number of feats 5000 or 10000
k=5#max ngram
v=1 #min mgram

# Vectorizing
liar_vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
X_train_liar = liar_vectorizer.fit_transform(tqdm.tqdm(liar_train)).toarray()
X_test_liar = liar_vectorizer.transform(tqdm.tqdm(liar_test)).toarray()
X_dev_liar = liar_vectorizer.transform(tqdm.tqdm(liar_dev)).toarray()
liar_feats = ['_'.join(s.split()) for s in liar_vectorizer.get_feature_names()]

kaggle_vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
X_train_kaggle = kaggle_vectorizer.fit_transform(tqdm.tqdm(kaggle_train)).toarray()
X_test_kaggle = kaggle_vectorizer.transform(tqdm.tqdm(kaggle_test)).toarray()
kaggle_feats = ['_'.join(s.split()) for s in kaggle_vectorizer.get_feature_names()]

# Classifiers
clf_liar=None
clf_liar = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(X_train_liar,liar_train_lab)
liar_coefs = clf_liar.coef_
allcoefs_liar = pd.DataFrame.from_records(liar_coefs, columns=liar_feats) #add ngrams as colnames
allcoefs_liar.to_csv('liar_coefs_final.csv', sep='\t', index=False)

clf_kaggle=None
clf_kaggle = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(X_train_kaggle,kaggle_train_lab)
kaggle_coefs = clf_kaggle.coef_
allcoefs_kaggle = pd.DataFrame.from_records(kaggle_coefs, columns=kaggle_feats) #add ngrams as colnames
allcoefs_kaggle.to_csv('kaggle_coefs_final.csv', sep='\t', index=False)


# get coefs from FakeNewsNet while your at it
FNN_vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
FNN_vect = FNN_vectorizer.fit_transform(FNN_X)
FNN_feats = ['_'.join(s.split()) for s in FNN_vectorizer.get_feature_names()]
clf_FNN=None
clf_FNN = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(FNN_vect,FNN_y)
FNN_coefs = clf_FNN.coef_
all_coefs_FNN = pd.DataFrame.from_records(FNN_coefs, columns=FNN_feats)
all_coefs_FNN.to_csv('FakeNewsNet_coefs_final.csv', sep='\t', index=False)


# Predicting
preds_liar_test = clf_liar.predict(X_test_liar)
preds_liar_valid = clf_liar.predict(X_dev_liar)
preds_liar_train = clf_liar.predict(X_train_liar)

preds_kaggle_test = clf_kaggle.predict(X_test_kaggle)
preds_kaggle_tain = clf_kaggle.predict(X_train_kaggle)

# cross-dataset predictions
kaggle_test_vectorized_by_liar = liar_vectorizer.transform(kaggle_test)
kaggle_test_predicted_by_liar_classifier = clf_liar.predict(kaggle_test_vectorized_by_liar)

kaggle_train_vectorized_by_liar = liar_vectorizer.transform(kaggle_train)
kaggle_train_predicted_by_liar = clf_liar.predict(kaggle_train_vectorized_by_liar)

# using other models to predict on FakeNewsNet
FNN_vectorized_by_liar = liar_vectorizer.transform(FNN_X)
FNN_predicted_by_liar = clf_liar.predict(FNN_vectorized_by_liar)

FNN_vectorized_by_kaggle = kaggle_vectorizer.transform(FNN_X)
FNN_predicted_by_kaggle = clf_kaggle.predict(FNN_vectorized_by_kaggle)



def print_scores(y, y_hat, string):
    print(string)
    print("binary F1", f1_score(y, y_hat))
    print("micro f1:",f1_score(y, y_hat, average='micro'))
    print("macro f1:", f1_score(y, y_hat, average='macro'))
    print("weighted F1", f1_score(y, y_hat, average='weighted'))
    print("accuracy", accuracy_score(y, y_hat))
    print()


print_scores(liar_test_lab, preds_liar_test, "Liar Test Scores")
print_scores(liar_dev_lab, preds_liar_valid, "Liar Valid. Scores")
print_scores(liar_train_lab, preds_liar_train, "Liar Train Scores")

print_scores(kaggle_test_lab, preds_kaggle_test, "Kaggle Test Scores")
print_scores(kaggle_train_lab, preds_kaggle_tain, "Kaggle Train Scores")

print_scores(kaggle_test_lab, kaggle_test_predicted_by_liar_classifier, "Kaggle Test Set Predicted by Classifier Trained on Liar")
print_scores(kaggle_train_lab, kaggle_train_predicted_by_liar, "Kaggle Train Set Predicted by Classifier Trained on Liar")

print_scores(FNN_y, FNN_predicted_by_liar, "FakeNewsNet Predicted by Classifier Trained on Liar")
print_scores(FNN_y, FNN_predicted_by_kaggle, "FakeNewsNet Predicted by Classifier Trained on Kaggle")

# random liar test set
random_liar_labels = liar_test_lab.copy()
np.random.shuffle(random_liar_labels)
print_scores(random_liar_labels,preds_liar_test, "Scores between predictions on Liar test set and randomized 'true' test labels. Classifier trained on Liar trains set")


print("Done")
