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
from get_liar_binary_data import load_liar_data


politifact_X_file = "data/politifact_X.txt"
politifact_y_file = "data/politifact_y.txt"
snopes_X_file = "data/snopes_X.txt"
snopes_y_file = "data/snopes_y.txt"

random.seed(16)
np.random.seed(16)

m= 10000 #number of feats 5000 or 10000
k=5#max ngram
v=1 #min mgram

def get_data(filenameX, filenamey):
    data = codecs.open(filenameX, 'r', 'utf-8').read().split('\n')
    data = [s.lower() for s in data if len(s) > 1]
    labels = codecs.open(filenamey, 'r', 'utf-8').read().split('\n')
    labels = [s for s in labels if len(s) > 1]
    return data, labels


snopes_X, snopes_y = get_data("data/snopes_X.txt", "data/snopes_y.txt")
politifact_X, politifact_y = get_data("data/politifact_X.txt", "data/politifact_y.txt")

snopes_y = [1 if i=="true" else 0 for i in snopes_y]

le = preprocessing.LabelEncoder()
politifact_y = le.fit_transform(politifact_y)
print(le.classes_) #['False' 'Half-True' 'Mostly False' 'Mostly True' 'Pants on Fire!' 'True']
print(le.transform(le.classes_)) # [0 1 2 3 4 5]
# untrue classes (to be encoded as 0): 0, 2, 4
# true classes (to be encoded as 1): 1, 3, 5

def binarize_label(labels):
    labels_transformed = [1 if i in [1,3,5] else 0 for i in labels]
    return labels_transformed

politifact_y = binarize_label(politifact_y)

def print_scores(y, y_hat, string):
    print(string)
    print("binary F1", f1_score(y, y_hat))
    print("micro f1:",f1_score(y, y_hat, average='micro'))
    print("macro f1:", f1_score(y, y_hat, average='macro'))
    print("weighted F1", f1_score(y, y_hat, average='weighted'))
    print("accuracy", accuracy_score(y, y_hat))
    print()

pol_X_train, pol_X_test, pol_y_train, pol_y_test = train_test_split(politifact_X, politifact_y, test_size=0.33, random_state=42)
sno_X_train, sno_X_test, sno_y_train, sno_y_test = train_test_split(snopes_X, snopes_y, test_size=0.33, random_state=42)
liar_train, liar_dev, liar_test, liar_train_lab, liar_dev_lab, liar_test_lab = load_liar_data()

def split_and_logreg(X_train, X_test, y_train, y_test, string_identifier, makecsv=True):
    print("train size:", len(y_train), "test size", len(y_test))
    vectorizer= None
    vectorizer = TfidfVectorizer(ngram_range=(v,k), max_features=m)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    feats = ['_'.join(s.split()) for s in vectorizer.get_feature_names()]
    clf = None
    clf = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000).fit(X_train,y_train)
    coefs = clf.coef_
    allcoefs = pd.DataFrame.from_records(coefs, columns=feats) #add ngrams as colnames
    if makecsv==True:
        allcoefs.to_csv(string_identifier+'.csv', sep='\t', index=False)
    predictions = clf.predict(X_test)
    print_scores(y_test, predictions, string_identifier)
    return

split_and_logreg(pol_X_train, pol_X_test, pol_y_train, pol_y_test, "politifact_coefs", makecsv=True)
print("first done")
split_and_logreg(sno_X_train, sno_X_test, sno_y_train, sno_y_test, "snopes_coefs", makecsv=True)
print("second done")
split_and_logreg(liar_train, pol_X_test, liar_train_lab, pol_y_test, "Politifact predicted by Liar classifier", makecsv=False)
print("third done")
split_and_logreg(liar_train, sno_X_test, liar_train_lab, sno_y_test, "Snopes predicted by Liar classifier", makecsv=False)

print("done")
