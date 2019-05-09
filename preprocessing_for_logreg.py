import numpy as np
import pandas as pd
import json
import os


#liar_train = pd.read_csv("/Users/Terne/Documents/KU/Speciale/Liar/liar_dataset/train.tsv", sep="\t", header=None)
#liar_val = pd.read_csv("/Users/Terne/Documents/KU/Speciale/Liar/liar_dataset/valid.tsv", sep="\t", header=None)
#liar_xtrain = liar_train[2]
#liar_ytrain = liar_train[1]
#liar_xval = liar_val[2]
#liar_yval = liar_val[1]
#datadir1 = "data/PolitiFact/FakeNewsContent"
#datadir2 = "data/PolitiFact/RealNewsContent"

politifact = pd.read_csv("data/PolitiFact/politifact.tsv", sep="\t", header=None)
politifact_X =politifact[4]
politifact_y = politifact[0]

snopes = pd.read_csv("data/Snopes/snopes.tsv", sep="\t", header=None)
snopes_X = snopes[3]
snopes_y = snopes[0]

def make_txt_file_from_json(datadir, savefile_name):
    with open(str(savefile_name), "w+") as file:
        for filename in os.listdir(datadir):
            with open(datadir+"/"+filename, 'r') as f:
                datastore = json.load(f)
                text = datastore["text"]
                file.write(text.replace("\n", ""))
                file.write("\n")


def make_txt_file_from_pandas_textcolumn(dataframe_column, savefile_name):
    with open(str(savefile_name), "w+") as file:
        for item in dataframe_column:
            file.write(str(item).replace("\n", ""))
            file.write("\n")

def save_labels_in_txt(dataframe_column, savefile_name):
    with open(str(savefile_name), "w+") as file:
        for label in dataframe_column:
            file.write(str(label))
            file.write("\n")

#make_txt_file_from_pandas_textcolumn(liar_xval, "liar_xval.txt")
#save_labels_in_txt(liar_yval, "liar_yval.txt")
#tr = codecs.open("liar_Xtrain", 'r', 'utf-8').read().split('\n')
#make_txt_file_from_json(datadir2, "RealNewsContent.txt")
make_txt_file_from_pandas_textcolumn(politifact_X, "politifact_X.txt")
save_labels_in_txt(politifact_y,"politifact_y.txt")

make_txt_file_from_pandas_textcolumn(snopes_X, "snopes_X.txt")
save_labels_in_txt(snopes_y,"snopes_y.txt")
