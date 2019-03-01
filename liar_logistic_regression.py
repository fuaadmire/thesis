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


#tr = codecs.open("kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
tr = codecs.open("data/liar_xtrain.txt", 'r', 'utf-8').read().split('\n')
#print(len(tr))
tr = [s.lower() for s in tr if len(s) > 1] #NOTE: I cannot do this with the kaggle data!
# Instead, remove last element and accept af few empty elements, or go through the elements and match the
# empty ones with their labels and remove both. Do a thorough check of what is going on (find empty element
# and print corropsonding element in dataframe).
#print(len(tr))
trlab = codecs.open('data/liar_ytrain.txt', 'r', 'utf-8').read().split('\n')
trlab = [s for s in trlab if len(s) > 1]
te = codecs.open("data/liar_xtest.txt", 'r', 'utf-8').read().split('\n')
te = [s.lower() for s in te if len(s) > 1]
telab = codecs.open("data/liar_ytest.txt", 'r', 'utf-8').read().split('\n')
telab = [s for s in telab if len(s) > 1]

assert len(tr) == len(trlab)

assert len(te) == len(telab)

random.seed(16)
np.random.seed(16)

m=int(sys.argv[1])#int(sys.argv[1]) #number of feats 5000 or 10000
k=5#max ngram
v=1 #min mgram

print(m)

vectorizer = CountVectorizer(ngram_range=(v,k), max_features=m)
print('fitting X_tr')
X_tr = vectorizer.fit_transform(tqdm.tqdm(tr)).toarray()
print('done')
print('fitting X_te')
X_te = vectorizer.transform(tqdm.tqdm(te)).toarray()
print('done')

feats = ['_'.join(s.split()) for s in vectorizer.get_feature_names()] #de m ngrams modellen bruger

print("fitting log reg")
clf=None
clf = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=1000, multi_class="multinomial").fit(X_tr, trlab)
print("done")
#print(clf.intercept_)
y_hat = clf.predict(X_te)
score=f1_score(te_lab, y_hat) #test accuracy er egentlig mindre vigtigt - det handler bare om at fitte. Det er dog meget smart så man kan se, at modellen lærer noget fornuftigt.
print(score)
allcoefs = pd.DataFrame.from_records(clf.coef_.tolist()[0], columns=feats) #add ngrams as colnames

allcoefs.to_csv('allcoefs_'+str(m)+'-'+str(k)+'gram-l1_+'.csv', sep='\t', index=False)
