


liar_train = pd.read_csv("Liar/liar_dataset/train.tsv", sep="\t", header=None)
liar_test = pd.read_csv("Liar/liar_dataset/test.tsv", sep="\t", header=None)
liar_xtrain = liar_train[2]
liar_ytrain = liar_train[1]
liar_xtest = liar_test[2]
liar_ytest = liar_test[1]

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

make_txt_file_from_pandas_textcolumn(liar_train, "liar_xtrain.txt")

#tr = codecs.open("liar_Xtrain", 'r', 'utf-8').read().split('\n')
