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

from os.path import join
from datetime import datetime
from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional, Masking
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import model_from_json


ID_WORD_EMBEDDING = 0
ID_POST_ID = 1
ID_CHUNK_ID = 2
ID_TAG_ID = 3


def load_pos_chunk(data_path):
    pos_path = os.path.join(data_path, "pos_data.pkl")
    chunk_path = os.path.join(data_path, "chunk_data.pkl")

    if os.path.isfile(pos_path) and os.path.isfile(chunk_path):
        pos_data = pkl.load(open(pos_path, "rb"))
        chunk_data = pkl.load(open(chunk_path, "rb"))
    else:
        pos_data, chunk_data = None, None
    return pos_data, chunk_data


class NameEntityRecognition:
    def __init__(self, model_path, words_path, embedding_vectors_path, tag_path, data_path):
        self.preprocessor = VnCoreNLP(address="http://127.0.0.1", port=9000)
        self.model = None
        self.load_model(model_path)
        self.utils = Utils(words_path, embedding_vectors_path, tag_path, *load_pos_chunk(data_path))
        self.r = Regex()

    def predict(self, data):
        try:
            data = str(data)
        except:
            data = str(data, encoding='utf-8')
        data = unicodedata.normalize('NFKC', data)
        pre_process_data = self.preprocessor.annotate(data)
        sentences = pre_process_data['sentences']
        word_list = []
        word_list_raw = []
        pos_list = []
        chunk_list = []
        for i, sen in enumerate(sentences):
            word_raw = [w['form'] for w in sen]
            pos_tag = [w['posTag'] for w in sen]
            words = list(map(lambda w: self.r.map_word_label(w), word_raw))
            chunks = list(map(lambda w: self.r.run_ex(w), words))
            word_list.append(words)
            word_list_raw.append(word_raw)
            pos_list.append(pos_tag)
            chunk_list.append(chunks)

        pos_id_list, alphabet_pos = self.utils.map_string_2_id_open(pos_list, 'pos')
        chunk_id_list, alphabet_chunk = self.utils.map_string_2_id_open(chunk_list, 'chunk')
        X = self.utils.create_vector_data_ex(word_list, pos_id_list, chunk_id_list)

        labels = self.model.predict_classes(X)
        print(labels)

        result = []
        for i in range(len(word_list_raw)):
            sen = []
            for j in range(len(word_list_raw[i])):
                label = self.utils.alphabet_tag.get_instance(labels[i][j])
                if label == None:
                    label = self.utils.alphabet_tag.get_instance(labels[i][j] + 1)
                if label == None:
                    raise ValueError('labels %s not define' % (labels[i][j]))
                sen.append((word_list_raw[i][j], label))
            result.append(sen)

        return self.get_final_result(result)

    def get_final_result(self, data):
        result = []
        for sen in data:
            s = []
            B_not_end = False
            for i, w in enumerate(sen):
                if u'B-' in w[1]:
                    if B_not_end:
                        s.append(u'</' + w[1].split(u'-')[1] + u'>')
                    s.append(u'<' + w[1].split(u'-')[1] + u'>')
                    s.append(w[0])
                    if i == len(sen) - 1:
                        s.append(u'</' + w[1].split(u'-')[1] + u'>')
                    B_not_end = True
                elif u'I-' in w[1] and i < len(sen) - 1 and sen[i+1][1] != w[1]:
                    s.append(w[0])
                    s.append(u'</' + w[1].split(u'-')[1] + u'>')
                    if B_not_end:
                        B_not_end = False
                elif u'I-' in w[1] and i == len(sen) - 1:
                    s.append(w[0])
                    s.append(u'</' + w[1].split(u'-')[1] + u'>')
                    if B_not_end:
                        B_not_end = False
                else:
                    if i != 0 and u'B-' in sen[i-1][1] and u'I-' not in w[1]:
                        s.append(u'</' + sen[i-1][1].split(u'-')[1] + u'>')
                        if B_not_end:
                            B_not_end = False
                    s.append(w[0])
            result.append(u' '.join(s))
        return u'\n'.join(result)

    def load_model(self, model_path):
        print('loading %s ...' % (model_path))
        if os.path.isdir(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None

