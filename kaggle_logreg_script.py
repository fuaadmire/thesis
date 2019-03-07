import pandas as pd
import tqdm
import numpy as np
import codecs
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import sys

random.seed(16)
np.random.seed(16)

m=int(sys.argv[1]) #number of feats 5000 or 10000
k=5#max ngram
v=1 #min mgram

print(m)


data = codecs.open("data/kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
data = data[:20800]
data = [s.lower() for s in data]

labels = codecs.open("data/kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
labels = labels[:20800]

tr, te, trlab, telab = train_test_split(data, labels, test_size=0.33, random_state=42)

# encode labels?


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
clf = LogisticRegression(random_state=16, solver='saga', penalty='l1', max_iter=1000).fit(X_tr, trlab) #terne, du skal skrive dit træningssæt her
print("done")

coefs = clf.coef_



with open("coefs.txt", "w+") as file:
    for c, f in zip(coefs[0],feats):
        file.write(f+"\t"+str(c)+"\n")

#allcoefs = pd.DataFrame.from_records(clf.coef_.tolist()[0], columns=feats) #add ngrams as colnames

#allcoefs.to_csv('allcoefs_'+str(m)+'-'+str(k)+'gram-l1_'+'.csv', sep='\t', index=False)
y_hat = clf.predict(X_te)
score=f1_score(telab, y_hat) #test accuracy er egentlig mindre vigtigt - det handler bare om at fitte. Det er dog meget smart så man kan se, at modellen lærer noget fornuftigt.
print(score)
