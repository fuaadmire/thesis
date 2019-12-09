#Installing keras-bert and keras adapter
#!pip install -q keras-bert keras-rectified-adam
#!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#!unzip -o uncased_L-12_H-768_A-12.zip

# my own data
from my_data_utils import load_liar_data, tile_reshape, load_kaggle_data, load_BS_data, load_TP_US_sample, load_TP_data_all_vs_us, load_TP_data_one_vs_us
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import codecs
import tensorflow as tf
from tqdm import tqdm
from chardet import detect
import keras
from keras_radam import RAdam
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import os
import numpy as np
import sys
import time
from tensorflow import set_random_seed
from keras import backend as K
import random


# Parameters
SEQ_LEN = 100
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4


def make_dev_from_train(train, train_lab):
    dev = train[int(abs((len(train_lab)/3)*2)):]
    dev_lab = train_lab[int(abs((len(train_lab)/3)*2)):]
    train = train[:int(abs((len(train_lab)/3)*2))]
    train_lab = train_lab[:int(abs((len(train_lab)/3)*2))]
    print(len(train), len(dev))
    return dev, dev_lab, train, train_lab

def test_f(model, tokenizer, test_string):
    global SEQ_LEN
    global trainingdata
    #global tokenizer
    print("Testing on "+test_string)
    if test_string == "kaggle":
        _, test, _, test_lab = load_kaggle_data(datapath)
    elif test_string == "BS":
        _, test, _, test_lab = load_BS_data(datapath)
    elif test_string == "liar":
        _, _, test, _, _, test_lab = load_liar_data(datapath)
    test_lab = to_categorical(test_lab, 2)
    test_indices = []
    for i in test:
        ids, segments = tokenizer.encode(i, max_len=SEQ_LEN)
        test_indices.append(ids)
    preds = model.predict([np.array(test_indices), np.zeros_like(test_indices)], verbose=0)
    print("len "+test_string+" preds:", len(preds))
    print("len "+test_string+" y_test", len(test_lab))
    np.savetxt("BERT"+trainingdata+"_"+test_string+"_labels.txt",test_lab)
    np.savetxt("BERT"+trainingdata+"_"+test_string+"_preds.txt",preds)
    print(preds)
    print(test_string+" accuracy: ",accuracy_score(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1)))
    print(test_string+" F1 score: ",f1_score(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1), average="weighted"))
    tn, fp, fn, tp = confusion_matrix(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1)).ravel()
    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)

def test_on_learnerdata(model, tokenizer):
    global SEQ_LEN
    global trainingdata
    prof_test = codecs.open(datapath+"proficiency/fce_text_entire_docs.txt", "r", "utf-8").read().split("\n")
    prof_test = prof_test[:len(prof_test)-1]
    test = [i.lower() for i in prof_test]
    test_lab = codecs.open(datapath+"proficiency/proficiency_entire_docs.txt", "r", "utf-8").read().split("\r\n")
    test_lab = test_lab[:len(test_lab)-1]
    test_indices = []
    for i in test:
        ids, segments = tokenizer.encode(i, max_len=SEQ_LEN)
        test_indices.append(ids)
    test_preds = model.predict([np.array(test_indices), np.zeros_like(test_indices)], verbose=0)
    print(len(test_preds))
    np.savetxt("BERT"+trainingdata+"_student_preds_softmax_entire_docs.txt",test_preds)


def test_on_TP(model, tokenizer):
    global SEQ_LEN
    global trainingdata
    lang_files = ["TP/da.test.txt", "TP/de.test.txt", "TP/es.test.txt", "TP/fr.test.txt",
                  "TP/it.test.txt", "TP/nl.test.txt", "TP/se.test.txt"]
    for i in lang_files:
        test, test_lab = load_TP_data_one_vs_us(datapath, i)
        test = [t.lower() for t in test]
        test_indices = []
        for t in test:
            ids, segments = tokenizer.encode(t, max_len=SEQ_LEN)
            test_indices.append(ids)
        test_preds = model.predict([np.array(test_indices), np.zeros_like(test_indices)], verbose=0)
        print(len(test_preds))
        np.savetxt("BERT"+trainingdata+"_"+i[3:5]+"_vs_us_"+"preds.txt",test_preds)
        np.savetxt("BERT"+trainingdata+"_"+i[3:5]+"_vs_us_"+"labels.txt",test_lab, fmt="%s")
    test, test_lab = load_TP_data_all_vs_us(datapath)
    test = [t.lower() for t in test]
    test_indices = []
    for i in test:
        ids, segments = tokenizer.encode(i, max_len=SEQ_LEN)
        test_indices.append(ids)
    test_preds = model.predict([np.array(test_indices), np.zeros_like(test_indices)], verbose=0)
    print(len(test_preds))
    np.savetxt("BERT"+trainingdata+"_TP_all_vs_us_"+"preds.txt",test_preds)
    np.savetxt("BERT"+trainingdata+"_TP_all_vs_us_"+"labels.txt",test_lab, fmt="%s")


for seed in [2]:#[2, 16, 42, 1, 4]:
    K.clear_session()
    model = None
    print("--------------------------------------")
    print("------------RANDOM SEED:",seed,"------")


    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

    starttime = time.time()
    # TF_KERAS must be added to environment variables in order to use TPU
    #os.environ['TF_KERAS'] = '1'


    # Pretrained model path
    pretrained_path = 'uncased_L-12_H-768_A-12'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')



    model = load_trained_model_from_checkpoint(
            config_path,
            checkpoint_path,
            training=True,
            trainable=True,
            seq_len=SEQ_LEN,
    )


    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = Tokenizer(token_dict)

    datapath = "/home/ktj250/thesis/data/"
    trainingdata = sys.argv[1] #"liar" # kaggle, FNC, BS
    print("trainingdata=",trainingdata)

    if trainingdata == "liar":
        train, dev, test, train_lab, dev_lab, test_lab = load_liar_data(datapath)
    elif trainingdata == "kaggle":
        train, test, train_lab, test_lab = load_kaggle_data(datapath)
    elif trainingdata == "BS":
        train, test, train_lab, test_lab = load_BS_data(datapath)

    train = [i.lower() for i in train]
    test = [i.lower() for i in test]



    if trainingdata == "liar":
        dev = [i.lower() for i in dev]
    else:
        dev, dev_lab, train, train_lab = make_dev_from_train(train, train_lab)

    train_indices = []
    test_indices = []
    dev_indices = []

    for i in train:
        ids, segments = tokenizer.encode(i, max_len=SEQ_LEN)
        train_indices.append(ids)

    for i in test:
        ids, segments = tokenizer.encode(i, max_len=SEQ_LEN)
        test_indices.append(ids)

    for i in dev:
        ids, segments = tokenizer.encode(i, max_len=SEQ_LEN)
        dev_indices.append(ids)

    train_lab = to_categorical(train_lab, 2)
    test_lab = to_categorical(test_lab, 2)
    dev_lab = to_categorical(dev_lab, 2)

    assert len(train_lab) == len(train_indices)


    opt = Adam(lr=LR)
    #opt = RAdam(learning_rate=LR)
    inputs = model.inputs[:2]
    dense = model.get_layer("NSP-Dense").output
    outputs = keras.layers.Dense(2, activation='softmax')(dense)
    model = keras.models.Model(inputs, outputs)
    model.compile(
            opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
    )

    # not sure whether this is needed or why?
    sess = K.get_session()
    uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
    init_op = tf.variables_initializer(
        [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
    )
    sess.run(init_op)

    model.fit([np.array(train_indices), np.zeros_like(train_indices)], train_lab, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    preds = model.predict([np.array(test_indices), np.zeros_like(test_indices)], verbose=0)
    print("len preds:", len(preds))
    np.savetxt("BERT"+trainingdata+"_"+trainingdata+"_labels.txt",test_lab)
    np.savetxt("BERT"+trainingdata+"_"+trainingdata+"_preds.txt",preds)
    print("len y_test", len(test_lab))
    print(preds)
    print("Accuracy: ",accuracy_score(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1)))
    print("F1 score: ",f1_score(np.argmax(test_lab,axis=1), np.argmax(preds, axis=1), average="weighted"))

    middletime = time.time()
    print("Time taken to train and test first part: ", middletime-starttime)



    if trainingdata == "liar":
        test_f(model, tokenizer, "kaggle")
        test_f(model, tokenizer, "BS")
    elif trainingdata == "kaggle":
        test_f(model, tokenizer, "liar")
        test_f(model, tokenizer, "BS")
    elif trainingdata == "BS":
        test_f(model, tokenizer, "liar")
        test_f(model, tokenizer, "kaggle")

    test_on_learnerdata(model, tokenizer)
    test_on_TP(model, tokenizer)

    print("Done.")
    endtime = time.time()
    print(endtime-starttime)
