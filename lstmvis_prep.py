import numpy as np


# datafile

# vocab_size: number of tokens in vocabulary
# max_doc_length: length of documents after padding (in Keras, the length of documents are usually padded to be of the same size)
# num_cells: number of LSTM cells
# num_samples: number of training/testing data samples
# num_time_steps: number of time steps in LSTM cells, usually equals to the size of input, i.e., max_doc_length
# trainTextsSeq: List of input sequence for each document (A matrix with size num_samples * max_doc_length)
# y_train: vector of document class labels
