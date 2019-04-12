import numpy as np
from keras.preprocessing import sequence
from keras.layers import Embedding, Input, Dense, LSTM, TimeDistributed
from keras.models import Model
from keras.layers import Dropout

from keras.constraints import NonNeg
from keras import regularizers


# prøv at se på output og weights med og uden nonneg constraint
# i Dense: kernel_constraint=non_neg()
# se også med regularizer, i Dense:
# fx kernel_regularizer=regularizers.l1(0.01)

X = np.random.randint(51, size=(100, 10)) # array of lists of same lenghts with random numbers from 0-50
y = np.random.randint(2, size=100)

def tile_reshape(train_lab, num_time_steps):
    y_train_tiled = np.tile(train_lab, (num_time_steps,1)).T
    y_train_tiled = y_train_tiled.reshape(len(train_lab), num_time_steps , 1)
    #print("y_train_shape:",y_train_tiled.shape)
    return y_train_tiled

y = tile_reshape(y, 10)

max_doc_length = 10
vocab_size = 51
embedding_size = 10
num_cells = 2
num_epochs = 10
num_batch = 16


myInput = Input(shape=(max_doc_length,), name='input')
x = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_doc_length)(myInput)
lstm_out = LSTM(num_cells, dropout=0.4, recurrent_dropout=0.4, return_sequences=True)(x)
predictions = TimeDistributed(Dense(1, bias=0, activation='sigmoid', kernel_constraint=NonNeg()))(lstm_out)
# ved ikke om jeg skal tilføje:
# activity_regularizer=regularizers.l1(0.01)
# eller:
# kernel_regularizer=regularizers.l1(0.01)
model = Model(inputs=myInput, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("fitting model..")
model.fit(X, y, epochs=num_epochs, verbose=2, batch_size=num_batch)
model.summary()
# how to print weights?
w = model.get_weights()
print(w)
print() #alle vægte for alle (4?) lag.
layer_w = model.layers[3].get_weights()[0] # Vægte fra sidste lag. De er positive efter kernel_constraint=NonNeg().
print(layer_w)
