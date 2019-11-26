#Installing keras-bert and keras adapter
#!pip install -q keras-bert keras-rectified-adam
#!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#!unzip -o uncased_L-12_H-768_A-12.zip

# my own data
from my_data_utils import load_liar_data, tile_reshape, load_kaggle_data, load_BS_data, load_TP_US_sample, load_TP_data_all_vs_us, load_TP_data_one_vs_us

import codecs
import tensorflow as tf
from tqdm import tqdm
from chardet import detect
import keras
from keras_radam import RAdam
from keras.optimizers import Adam
from keras import backend as K
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import os
import numpy as np
import sys

# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'

# Parameters
SEQ_LEN = 100
BATCH_SIZE = 64
EPOCHS = 7
LR = 1e-4

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

train = [i.lower() for i in train]
test = [i.lower() for i in test]

if trainingdata == "liar":
    dev = [i.lower() for i in dev]
else:
    dev = train[int(abs((len(train_lab)/3)*2)):]
    dev_lab = train_lab[int(abs((len(train_lab)/3)*2)):]
    train = train[:int(abs((len(train_lab)/3)*2))]
    train_lab = train_lab[:int(abs((len(train_lab)/3)*2))]
    print(len(train), len(dev))

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

opt = Adam(lr=LR)
#opt = RAdam(learning_rate=LR)

inputs = model.inputs[:2]
dense = model.get_layer("NSP-Dense").output
outputs = keras.layers.Dense(2, activation='softmax')(dense)
model.keras.models.Model(inputs, outputs)
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

model.fit(train_indices, train_lab, epochs=EPOCHS, batch_size=BATCH_SIZE)

preds = model.predict(test_indices, verbose=True).argmax(axis=-1)
print(np.mean([preds==test_lab]))
print(preds)
