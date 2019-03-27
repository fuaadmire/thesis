import nltk
import re
import numpy as np

def preprocess(data):
    remove_punct = [re.sub(r'[^\w\s]', '', " ".join(i)) for i in data]
    #print(remove_punct)
    tokens = [nltk.word_tokenize(i.lower()) for i in remove_punct]
    #print(tokens)
    wnl = nltk.WordNetLemmatizer()
    lemmas = [[wnl.lemmatize(t) for t in tok] for tok in tokens]
    #print(len(lemmas[0]))
    return lemmas

#data = [["hey what's up beautiful? Is'nt, it lovely houses."], ["wow such spirit and fruitfulness in the apples"]]

#lemmas = preprocess(data)

#print(lemmas)
