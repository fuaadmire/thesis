import pandas as pd
import tqdm
import numpy as np
import codecs
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import random
import matplotlib.pyplot as plt
import sys


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

random.seed(16)
np.random.seed(16)

m= 10000 #number of feats 5000 or 10000
k=5#max ngram
v=1 #min mgram

print("LIAR")
vectorizer = CountVectorizer(ngram_range=(v,k), max_features=m)
print('fitting Liar X_train')
X_train_liar = vectorizer.fit_transform(tqdm.tqdm(liar_train)).toarray()
print('done')
print('fitting X_test')
X_test_liar = vectorizer.transform(tqdm.tqdm(liar_test)).toarray()
print('done')

feats = ['_'.join(s.split()) for s in vectorizer.get_feature_names()] #de m ngrams modellen bruger

print("fitting log reg")
clf=None
clf = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000, multi_class="multinomial").fit(X_train_liar, liar_train_lab)
print("done")
#print(clf.intercept_)

coefs = clf.coef_

allcoefs = pd.DataFrame.from_records(coefs, columns=feats) #add ngrams as colnames

allcoefs.to_csv('liar_TEST_coefs_'+str(m)+'feats'+'_'+str(k)+'gram-l1'+'.csv', sep='\t', index=False)
print("classes:", clf.classes_)

y_hat = clf.predict(X_test_liar)
microf1=f1_score(liar_test_lab, y_hat, average='micro')
print("MicroF1", microf1)
macrof1=f1_score(liar_test_lab, y_hat, average='macro')
print("macroF1", macrof1)
print("accuracy", np.mean([y_hat == liar_test_lab]))
print("weighted F1", f1_score(liar_test_lab, y_hat, average='weighted'))
