

def feature_model(seq,
                    test_seq,
                    train_lab,
                    test_lab,
                    embedding_matrix,
                    vocab_size,
                    dropout,
                    r_dropout,
                    num_cells,
                    learning_rate,
                    num_epochs,
                    trainingdata,
                    max_doc_length=100,
                    embedding_size=300,
                    TIMEDISTRIBUTED=False,
                    dev_seq=None,
                    dev_lab=None,
                    datapath="/home/ktj250/thesis/data/"):


    if trainingdata == "liar":
        train_tags, test_tags, dev_tags = load_pos_tags(trainingdata, datapath)
    else:
        train_tags, test_tags = load_pos_tags(trainingdata, datapath)


    tags = set([])
    for ts in train_tags:
        for t in ts:
            tags.add(t)
    tag2id = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2id["UNK"] = len(tag2id)+1

    # save the dict for later

    pos_seq = np.array([[tag2id[t] for t in ts] for ts in train_tags])
    test_pos_seq = np.array([[tag2id[t] for t in ts] for ts in test_tags])
    if trainingdata=="liar":
        dev_pos_seq = np.array([[tag2id[t] for t in ts] for ts in dev_tags])

    # pad tags
    pos_seq = sequence.pad_sequences(pos_seq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)
    test_pos_seq = sequence.pad_sequences(test_pos_seq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)
    if trainingdata=="liar":
        dev_pos_seq = sequence.pad_sequences(dev_pos_seq, maxlen=max_doc_length, dtype='int32', padding='post', truncating='post', value=0.0)


    text_input = Input(shape=(max_doc_length,), name='text input')
    pos_tag_input = Input(shape=(max_doc_length,), name='pos input')

    print(myInput.shape)
    if use_pretrained_embeddings:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix],input_length=max_doc_length,trainable=True)(text_input)
    else:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_doc_length)(text_input)
        print(x.shape)

    if TIMEDISTRIBUTED:
        lstm_out = LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True, kernel_constraint=NonNeg())(x)
        aux_output = TimeDistributed(Dense(1, activation='sigmoid', kernel_constraint=NonNeg()))(lstm_out)
        feats_concat = keras.layers.concatenate([lstm_out, pos_tag_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        predictions = TimeDistributed(Dense(1, activation='sigmoid', kernel_constraint=NonNeg()))(x)

    else:
        lstm_out = Bidirectional(LSTM(num_cells, dropout=dropout, recurrent_dropout=r_dropout))(x)
        aux_output = Dense(2, activation='softmax')(lstm_out)
        feats_concat = keras.layers.concatenate([lstm_out, pos_tag_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(lstm_out)

    model = Model(inputs=[text_input, pos_tag_input], outputs=[predictions,aux_output])

    opt = Adam(lr=learning_rate)

    if TIMEDISTRIBUTED:
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("fitting model..")
    if trainingdata == "liar":
        history = model.fit([seq, pos_seq], [train_lab,train_lab], epochs=num_epochs, verbose=2, batch_size=num_batch, validation_data=([dev_seq,dev_pos_seq],[dev_lab,dev_lab]))
    else:
        history = model.fit([seq, pos_seq], [train_lab,train_lab], epochs=num_epochs, verbose=2, batch_size=num_batch)
    model.summary()

    return model, history
