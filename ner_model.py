# -*- encoding: utf-8 -*-
import unicodedata
import subprocess
import shlex
import os
import argparse

import numpy as np
import tensorflow as tf
import pickle as pkl

from regex import Regex
from Utils import Utils
from gen_data import DataGenerator

from datetime import datetime
from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize

from sklearn.externals import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional, Masking
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.layers import CRF


ID_WORD_EMBEDDING = 0
ID_POST_ID = 1
ID_CHUNK_ID = 2
ID_TAG_ID = 3


def building_ner(num_lstm_layer, num_hidden_node, dropout,
                 time_step, vector_length, num_labels):
    model = Sequential()

    model.add(Masking(mask_value=0., input_shape=(time_step, vector_length)))
    for i in range(num_lstm_layer-1):
        model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True,
                                     dropout=dropout, recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True,
                                 dropout=dropout, recurrent_dropout=dropout),
                            merge_mode='concat'))
    # crf = CRF(output_lenght, sparse_target=False, name='CRF')
    # model.add(crf)
    # model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.add(TimeDistributed(Dense(num_labels)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_pre_data():

    print('Loading data...')
    utils = Utils(word_dir, vector_dir, tag_path)
    train_data, valid_data, test_data = utils.create_data(train_dir, dev_dir, test_dir)

    # self.utils.mkdir('model')
    # self.save_data(input_train, output_train, input_dev, output_dev, input_test, output_test)
    # joblib.dump(self.utils, 'model/utils.pkl')

    return train_data, valid_data, test_data


def get_test_data(test_data):
    embed_words = test_data[ID_WORD_EMBEDDING]
    pos_ids = test_data[ID_POST_ID]
    chunk_ids = test_data[ID_CHUNK_ID]
    labels = test_data[ID_TAG_ID]

    input_test = embed_words
    input_test = np.concatenate((input_test, pos_ids), axis=2)
    input_test = np.concatenate((input_test, chunk_ids), axis=2)

    output_test = labels
    return input_test, output_test


def load_pos_chunk(data_path="/home/trungtq/Documents/NER/vie-ner-lstm/python3_ver/model/data"):
    pos_path = os.path.join(data_path, "pos_data.pkl")
    chunk_path = os.path.join(data_path, "chunk_data.pkl")

    if os.path.isfile(pos_path) and os.path.isfile(chunk_path):
        pos_data = pkl.load(open(pos_path, "rb"))
        chunk_data = pkl.load(open(chunk_path, "rb"))
    else:
        pos_data, chunk_data = None, None
    return pos_data, chunk_data


class Network:
    def __init__(self, num_lstm_layers, num_hidden_nodes, dropout, time_step, vector_length, num_labels):
        self.num_lstm_layers = num_lstm_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.dropout = dropout
        self.time_step = time_step
        self.vector_length = vector_length
        self.num_labels = num_labels

    def build_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(self.time_step, self.vector_length)))
        for i in range(num_lstm_layer - 1):
            model.add(Bidirectional(LSTM(units=self.num_hidden_nodes, return_sequences=True,
                                         dropout=self.dropout, recurrent_dropout=self.dropout)))
        model.add(Bidirectional(LSTM(units=self.num_hidden_nodes, return_sequences=True,
                                     dropout=self.dropout, recurrent_dropout=self.dropout),
                                merge_mode='concat'))
        # crf = CRF(output_lenght, sparse_target=False, name='CRF')
        # model.add(crf)
        # model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        model.add(TimeDistributed(Dense(self.num_labels)))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


words_path = "/home/trungtq/Documents/NER/data/words.pl"
embedding_vectors_path = "/home/trungtq/Documents/NER/data/vectors.npy"
tag_path = "/home/trungtq/Documents/NER/vie-ner-lstm/python3_ver/model/data/tag_data.pkl"


class NameEntityRecognition:
    def __init__(self):
        self.preprocessor = VnCoreNLP(address="http://127.0.0.1", port=9000)
        self.model = None
        self.utils = Utils(words_path, embedding_vectors_path, tag_path, *load_pos_chunk())
        self.re = Regex()

    def build_model(self, num_lstm_layer, num_hidden_node, dropout, batch_size, patience):

        startTime = datetime.now()

        train_data, valid_data, test_data = get_pre_data()
        train_gen = DataGenerator(train_data, batch_size=batch_size)
        valid_gen = DataGenerator(valid_data, batch_size=batch_size)

        print('Building model...')
        time_step = train_data[ID_WORD_EMBEDDING].shape[1]

        input_length = train_data[ID_WORD_EMBEDDING].shape[2] + train_data[ID_POST_ID].shape[2] + train_data[ID_CHUNK_ID].shape[2]
        output_length = np.shape(train_data[ID_TAG_ID])[2]

        self.network = Network(num_lstm_layer, num_hidden_node,
                               dropout, time_step, input_length,
                               output_length)
        self.model = self.network.build_model()
        print('Model summary...')
        print(self.model.summary())

        print('Training model...')
        early_stopping = EarlyStopping(patience=patience)
        ckpt = tf.keras.callbacks.ModelCheckpoint('ckpt/ner.ckpt',
                                                  save_weights_only=True,
                                                  save_freq=10,
                                                  verbose=1)

        self.model.fit_generator(generator=train_gen,
                                 validation_data=valid_gen,
                                 epochs=100,
                                 callbacks=[early_stopping, ckpt])

        self.model.save('model/ner_model')
        endTime = datetime.now()
        print("Running time: ")
        print(endTime - startTime)

        print("Start testing")
        x_test, y_test = get_test_data(test_data)
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"Result on test data: test loss {test_loss}, test_accuracy {test_acc}")

    def load_model(self, model_path):
        print('loading %s ...' % (model_path))
        if os.path.isdir(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None


if __name__ == '__main__':

    from os.path import join, dirname

    word_dir = join(dirname(__file__), "data/words.pl")
    vector_dir = join(dirname(__file__), "data/vectors.npy")
    train_dir = join(dirname(__file__), "data/new/normalize_data/train_sample.txt")
    dev_dir = join(dirname(__file__), "data/new/normalize_data/val_sample.txt")
    test_dir = join(dirname(__file__), "data/new/normalize_data/test_sample.txt")
    num_lstm_layer = 2
    num_hidden_node = 128
    dropout = 0.5
    batch_size = 64
    patience = 3

    n = NameEntityRecognition()
    n.build_model(num_lstm_layer, num_hidden_node, dropout, batch_size, patience)

    # n.load_model("/home/trungtq/Documents/NER/vie-ner-lstm/python3_ver/model/ner_model")
    # result = n.predict("Tôi tên là Trần Quang Trung. Địa chỉ nhà tại 1/2/3/4 Chính Kinh, Thanh Xuân, Hà Nội")

