import codecs
import numpy as np
from sklearn.model_selection import train_test_split
import re
import pickle
from sklearn.utils import shuffle



def binarize_labels(labels, FAKE):
    if FAKE==1:
        #labels_transformed = [0 if i in [2,3,5] else 1 for i in labels]
        labels_transformed = [0 if i in ['half-true','mostly-true','true'] else 1 for i in labels]
    else:
        #labels_transformed = [1 if i in [2,3,5] else 0 for i in labels]
        labels_transformed = [1 if i in ['half-true','mostly-true','true'] else 0 for i in labels]
    print("Training with Fake=",FAKE)
    return labels_transformed


def load_liar_data(datapath, FAKE=1):

    liar_train = codecs.open(datapath+"liar_xtrain.txt", 'r', 'utf-8').read().split('\n')
    liar_train = [s.lower() for s in liar_train if len(s) > 1]
    liar_train_labels = codecs.open(datapath+'liar_ytrain.txt', 'r', 'utf-8').read().split('\n')
    liar_train_lab = [s for s in liar_train_labels if len(s) > 1]

    liar_dev = codecs.open(datapath+"liar_xval.txt", 'r', 'utf-8').read().split('\n')
    liar_dev = [s.lower() for s in liar_dev if len(s) > 1]
    liar_dev_labels = codecs.open(datapath+"liar_yval.txt", 'r', 'utf-8').read().split('\n')
    liar_dev_lab = [s for s in liar_dev_labels if len(s) > 1]

    liar_test = codecs.open(datapath+"liar_xtest.txt", 'r', 'utf-8').read().split('\n')
    liar_test = [s.lower() for s in liar_test if len(s) > 1]
    liar_test_labels = codecs.open(datapath+"liar_ytest.txt", 'r', 'utf-8').read().split('\n')
    liar_test_lab = [s for s in liar_test_labels if len(s) > 1]

    assert len(liar_train) == len(liar_train_lab)
    assert len(liar_dev) == len(liar_dev_lab)
    assert len(liar_test) == len(liar_test_lab)

    # BINARIZE LABELS, IF FAKE=1 THEN THE UNTRUE CLASSES WILL BE LABELLED AS 1.
    liar_train_lab = binarize_labels(liar_train_lab, FAKE)
    liar_dev_lab = binarize_labels(liar_dev_lab, FAKE)
    liar_test_lab = binarize_labels(liar_test_lab, FAKE)

    return liar_train, liar_dev, liar_test, liar_train_lab, liar_dev_lab, liar_test_lab


def load_kaggle_data(datapath):
    print("Kaggle labels: \n 1: unreliable, 0: reliable")
    data = codecs.open(datapath+"kaggle_trainset.txt", 'r', 'utf-8').read().split('\n')
    data = data[:20800]
    data = [s.lower() for s in data]
    labels = codecs.open(datapath+"kaggle_train_labels.txt", 'r', 'utf-8').read().split('\n')
    labels = labels[:20800]
    labels = [int(i) for i in labels]
    # disregarding input which is less than 100 characters (as they do not contain many words, if any)
    #labels_include = []
    #data_include = []
    #for indel, i in enumerate(data):
    #    if len(i) > 100:
    #        data_include.append(i)
    #        labels_include.append(labels[indel])
    new_data, new_labels = remove_duplicates(data, labels)
    train, test, train_lab, test_lab = train_test_split(new_data, new_labels, test_size=0.33, random_state=42)
    # remove city names
    return train, test, train_lab, test_lab


def load_FNC_data(datapath):
    FNC_fake = codecs.open(datapath+"FNC_fake_part1.txt", 'r', 'utf-8').read().split('\n')
    FNC_fake = FNC_fake[:25000]
    FNC_fake = [s.lower() for s in FNC_fake]
    #print(FNC_fake[0])
    FNC_true = codecs.open(datapath+"FNC_true_part1.txt", 'r', 'utf-8').read().split('\n')
    FNC_true = FNC_true[:25000]
    FNC_true = [s.lower() for s in FNC_true]

    print("FNC labels: \n 1: Fake, 0: Reliable")
    FNC_fake_labels = np.ones(len(FNC_fake))
    FNC_true_labels = np.zeros(len(FNC_true))

    FNC_samples = np.concatenate((FNC_fake, FNC_true))
    FNC_labels = np.concatenate((FNC_fake_labels, FNC_true_labels))
    assert len(FNC_samples) == len(FNC_labels)
    FNC_samples, FNC_labels = shuffle(FNC_samples, FNC_labels, random_state=42)
    FNC_Xtrain, FNC_Xtest, FNC_ytrain, FNC_ytest = train_test_split(FNC_samples, FNC_labels, test_size=0.33, random_state=42)
    return FNC_Xtrain, FNC_Xtest, FNC_ytrain, FNC_ytest


def load_BS_data(datapath):
    fake = pickle.load(open(datapath+"BS_detector/fake.pkl", "rb"))
    fake = [s.strip().lower() for s in fake]
    real = pickle.load(open(datapath+"BS_detector/real.pkl", "rb"))
    real = [s.strip().lower() for s in real]
    samples = fake+real
    print("BS labels: 1=fake, 0=real")
    labels = [1 for _ in fake] + [0 for _ in real]
    assert len(samples) == len(labels)
    samples, labels = shuffle(samples, labels, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def load_TP_data(datapath):
    da = codecs.open("da.test.txt", "r", "utf-8").read().split("\n")
    da = da[:len(da)-1]
    de = codecs.open("de.test.txt", "r", "utf-8").read().split("\n")
    de = de[:len(de)-1]
    es = codecs.open("es.test.txt", "r", "utf-8").read().split("\n")
    es = es[:len(es)-1]
    fr = codecs.open("fr.test.txt", "r", "utf-8").read().split("\n")
    fr = fr[:len(fr)-1]
    it = codecs.open("it.test.txt", "r", "utf-8").read().split("\n")
    it = it[:len(it)-1]
    nl = codecs.open("nl.test.txt", "r", "utf-8").read().split("\n")
    nl = nl[:len(nl)-1]
    se = codecs.open("se.test.txt", "r", "utf-8").read().split("\n")
    se = se[:len(se)-1]
    

    pass


# Reshaping function for labels
def tile_reshape(train_lab, num_time_steps):
    y_train_tiled = np.tile(train_lab, (num_time_steps,1)).T
    y_train_tiled = y_train_tiled.reshape(len(train_lab), num_time_steps , 1)
    #print("y_train_shape:",y_train_tiled.shape)
    return y_train_tiled


def label_switch(labels):
    labels_transformed = [1 if i==0 else 0 for i in labels]
    return labels_transformed


def remove_duplicates(data, labels):
    new_data = []
    new_labels = []
    for i,j in zip(data, labels):
        if i not in new_data:
            new_data.append(i)
            new_labels.append(j)
    return new_data, new_labels


def clean_patterns():

    pass

def pos_tagging(trainingdata, datapath):
    import nltk

    if trainingdata == "liar":
        train, dev, test, train_lab, dev_lab, test_lab = load_liar_data(datapath)
    elif trainingdata == "kaggle":
        train, test, train_lab, test_lab = load_kaggle_data(datapath)
    elif trainingdata == "FNC":
        train, test, train_lab, test_lab = load_FNC_data(datapath)
    elif trainingdata == "BS":
        train, test, train_lab, test_lab = load_BS_data(datapath)

    train = [nltk.word_tokenize(i.lower()) for i in train]

    test = [nltk.word_tokenize(i.lower()) for i in test]

    if trainingdata == "liar":
        dev = [nltk.word_tokenize(i.lower()) for i in dev]

    with open(trainingdata+"_train_POS.txt", mode="w") as file:
        for line in train:
            pos_tags = nltk.pos_tag(line)
            pos_tag_string = " ".join([i[1] for i in pos_tags])
            file.write(pos_tag_string+"\n")
            #file.write("\n")

    with open(trainingdata+"_test_POS.txt", mode="w") as file:
        for line in test:
            pos_tags = nltk.pos_tag(line)
            pos_tag_string = " ".join([i[1] for i in pos_tags])
            file.write(pos_tag_string+"\n")

    if trainingdata == "liar":
        with open(trainingdata+"_dev_POS.txt", mode="w") as file:
            for line in dev:
                pos_tags = nltk.pos_tag(line)
                pos_tag_string = " ".join([i[1] for i in pos_tags])
                file.write(pos_tag_string+"\n")
    # save to txt file in same format, seperate by newline

def load_pos_tags(trainingdata, datapath):
    train_tags = codecs.open(datapath+"pos_tags/"+trainingdata+"_train_POS.txt").read().split('\n')
    train_tags = train_tags[:len(train_tags)-1]
    test_tags = codecs.open(datapath+"pos_tags/"+trainingdata+"_test_POS.txt").read().split('\n')
    test_tags = test_tags[:len(test_tags)-1]
    if trainingdata=="liar":
        dev_tags = codecs.open(datapath+"pos_tags/liar_dev_POS.txt").read().split('\n')
        dev_tags = dev_tags[:len(dev_tags)-1]
        return train_tags, test_tags, dev_tags
    else:
        return train_tags, test_tags
