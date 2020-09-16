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

from os.path import join, dirname
from datetime import datetime
from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize

from tf2crf import CRF
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional, Masking
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.layers import CRF

ID_WORD = 0
ID_POS = 1
ID_CHUNK = 2
ID_TAG = 3


def building_ner(num_lstm_layer, num_hidden_node, dropout,
                 time_step, vector_length, num_labels, is_crf=False):
    model = Sequential()

    model.add(Masking(mask_value=0., input_shape=(time_step, vector_length)))
    for i in range(num_lstm_layer-1):
        model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True,
                                     dropout=dropout, recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True,
                                 dropout=dropout, recurrent_dropout=dropout),
                            merge_mode='concat'))

    if is_crf:
        crf = CRF(units=num_labels)
        model.add(crf)
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
    else:
        model.add(TimeDistributed(Dense(num_labels)))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_pre_data(utils: Utils):

    print('Loading data...')
    train_data, valid_data, test_data = utils.create_data(train_dir, dev_dir, test_dir)

    # self.utils.mkdir('model')
    # self.save_data(input_train, output_train, input_dev, output_dev, input_test, output_test)
    # joblib.dump(self.utils, 'model/utils.pkl')

    return train_data, valid_data, test_data


def get_test_data(test_data, utils: Utils):
    word_list = test_data[ID_WORD]
    pos_list = test_data[ID_POS]
    chunk_list = test_data[ID_CHUNK]
    tag_list = test_data[ID_TAG]

    word_tensor, pos_tensor, chunk_tensor, tag_tensor = utils.create_vector_data(word_list,
                                                                                 pos_list,
                                                                                 chunk_list,
                                                                                 tag_list)
    input_test = np.concatenate((word_tensor, pos_tensor, chunk_tensor), axis=2)
    output_test = tag_tensor

    return input_test, output_test


def load_pos_chunk(data_path=join(dirname(__file__), "model/data")):
    pos_path = os.path.join(data_path, "pos_data.pkl")
    chunk_path = os.path.join(data_path, "chunk_data.pkl")

    if os.path.isfile(pos_path) and os.path.isfile(chunk_path):
        pos_data = pkl.load(open(pos_path, "rb"))
        chunk_data = pkl.load(open(chunk_path, "rb"))
    else:
        pos_data, chunk_data = None, None
    return pos_data, chunk_data


class Network:
    def __init__(self, num_lstm_layers, num_hidden_nodes, dropout, time_step, vector_length, num_labels, is_crf=True):
        self.num_lstm_layers = num_lstm_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.dropout = dropout
        self.time_step = time_step
        self.vector_length = vector_length
        self.num_labels = num_labels
        self.is_crf = is_crf

    def build_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(self.time_step, self.vector_length)))
        for i in range(num_lstm_layer - 1):
            model.add(Bidirectional(LSTM(units=self.num_hidden_nodes, return_sequences=True,
                                         dropout=self.dropout, recurrent_dropout=self.dropout)))
        model.add(Bidirectional(LSTM(units=self.num_hidden_nodes, return_sequences=True,
                                     dropout=self.dropout, recurrent_dropout=self.dropout),
                                merge_mode='concat'))
        model.add(TimeDistributed(Dense(self.num_labels)))
        if self.is_crf:
            crf = CRF(dtype='float32')
            model.add(crf)
            model.compile(optimizer='adam', loss=crf.loss, metrics=[crf.accuracy])
        else:
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


class NameEntityRecognition:
    def __init__(self):
        self.model = None
        self.utils = Utils(word_dir, vector_dir)
        self.r = Regex()

    def build_model(self, num_lstm_layer, num_hidden_node, dropout, batch_size, patience, n_jobs=6):

        startTime = datetime.now()

        train_data, valid_data, test_data = get_pre_data(self.utils)
        train_gen = DataGenerator(train_data, self.utils, batch_size=batch_size)
        valid_gen = get_test_data(valid_data, utils=self.utils)

        print('Building model...')
        time_step = self.utils.max_length

        input_length = self.utils.embedd_vectors.shape[1] + self.utils.alphabet_pos.size() + \
                       self.utils.alphabet_chunk.size()
        output_length = self.utils.alphabet_tag.size()

        self.network = Network(num_lstm_layer, num_hidden_node,
                               dropout, time_step, input_length,
                               output_length)

        # tf.config.threading.set_intra_op_parallelism_threads(n_jobs)
        tf.config.threading.set_inter_op_parallelism_threads(n_jobs)

        with tf.device('/CPU:0'):
            self.model = self.network.build_model()
            print('Model summary...')
            print(self.model.summary())

            print('Training model...')
            early_stopping = EarlyStopping(patience=patience)
            ckpt = tf.keras.callbacks.ModelCheckpoint('ckpt/ner.ckpt',
                                                      # save_weights_only=True,
                                                      save_best_only=True,
                                                      save_freq='epoch',
                                                      verbose=1
                                                      )

            tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", update_freq="batch")

            self.model.fit_generator(generator=train_gen,
                                     validation_data=valid_gen,
                                     epochs=100,
                                     max_queue_size=100,
                                     callbacks=[early_stopping, ckpt, tensorboard])

        self.model.save('model/ner_model')
        endTime = datetime.now()
        print("Running time: ")
        print(endTime - startTime)

        print("Start testing")
        x_test, y_test = get_test_data(test_data, utils=self.utils)
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"Result on test data: test loss {test_loss}, test_accuracy {test_acc}")

    def load_model(self, model_path):
        print('loading %s ...' % (model_path))
        if os.path.isdir(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None


if __name__ == '__main__':

    from os.path import join, dirname, abspath

    word_dir = join(dirname(abspath(__file__)), "data/words.pl")
    vector_dir = join(dirname(abspath(__file__)), "data/vectors.npy")
    train_dir = join(dirname(abspath(__file__)), "data/newz/normalized_data/train_sample.txt")
    dev_dir = join(dirname(abspath(__file__)), "data/newz/normalized_data/dev_sample.txt")
    test_dir = join(dirname(abspath(__file__)), "data/newz/normalized_data/test_sample.txt")
    # train_dir = join(dirname(abspath(__file__)), "data/small_data/train_sample.txt")
    # dev_dir = join(dirname(abspath(__file__)), "data/small_data/val_sample.txt")
    # test_dir = join(dirname(abspath(__file__)), "data/small_data/test_sample.txt")
    num_lstm_layer = 2
    num_hidden_node = 128
    dropout = 0.2
    batch_size = 64
    patience = 3

    n = NameEntityRecognition()
    n.build_model(num_lstm_layer, num_hidden_node, dropout, batch_size, patience)
