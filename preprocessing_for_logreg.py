import numpy as np
import pandas as pd


liar_train = pd.read_csv("/Users/Terne/Documents/KU/Speciale/Liar/liar_dataset/train.tsv", sep="\t", header=None)
liar_val = pd.read_csv("/Users/Terne/Documents/KU/Speciale/Liar/liar_dataset/valid.tsv", sep="\t", header=None)
liar_xtrain = liar_train[2]
liar_ytrain = liar_train[1]
liar_xval = liar_val[2]
liar_yval = liar_val[1]

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
save_labels_in_txt(liar_yval, "liar_yval.txt")
#tr = codecs.open("liar_Xtrain", 'r', 'utf-8').read().split('\n')
