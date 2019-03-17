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


print("liar logreg")

tr = codecs.open("data/liar_xtrain.txt", 'r', 'utf-8').read().split('\n')
#print(len(tr))
tr = [s.lower() for s in tr if len(s) > 1]
trlab = codecs.open('data/liar_ytrain.txt', 'r', 'utf-8').read().split('\n')
trlab = [s for s in trlab if len(s) > 1]


val = codecs.open("data/liar_xval.txt", 'r', 'utf-8').read().split('\n')
#print(len(val))
val = [s.lower() for s in val if len(s) > 1]
vallab = codecs.open("data/liar_yval.txt", 'r', 'utf-8').read().split('\n')
vallab = [s for s in vallab if len(s) > 1]




assert len(tr) == len(trlab)

assert len(val) == len(vallab)

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
X_val = vectorizer.transform(tqdm.tqdm(val)).toarray()
print('done')

feats = ['_'.join(s.split()) for s in vectorizer.get_feature_names()] #de m ngrams modellen bruger

print("fitting log reg")
clf=None
clf = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=10000, multi_class="multinomial").fit(X_tr, trlab)
print("done")
#print(clf.intercept_)

coefs = clf.coef_

allcoefs = pd.DataFrame.from_records(coefs, columns=feats) #add ngrams as colnames

allcoefs.to_csv('liar_coefs_'+str(m)+'feats'+'_'+str(k)+'gram-l1'+'.csv', sep='\t', index=False)
print("classes:", clf.classes_)

y_hat = clf.predict(X_val)
microf1=f1_score(vallab, y_hat, average='micro') #test accuracy er egentlig mindre vigtigt - det handler bare om at fitte. Det er dog meget smart så man kan se, at modellen lærer noget fornuftigt.
print("MicroF1", microf1)
macrof1=f1_score(vallab, y_hat, average='macro')
print("macroF1", macrof1)
print("accuracy", np.mean([y_hat == vallab]))
print("weighted F1", f1_score(vallab, y_hat, average='weighted'))
