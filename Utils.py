import codecs
import os

import numpy as np
import pickle as pkl

from os.path import join, dirname
from Alphabet import Alphabet
from regex import Regex


class Utils:
    def __init__(self, word_dir, vector_dir, alphabet_pos=None, alphabet_chunk=None):

        # load pre-train word2vec
        self.embedd_vectors = np.load(vector_dir)
        with open(word_dir, 'rb') as handle:
            self.embedd_words = pkl.load(handle)
        self.embedd_dim = np.shape(self.embedd_vectors)[1]
        # gen embedding vector for unknown token
        self.unknown_embedd = np.random.uniform(-0.01, 0.01, (1, self.embedd_dim))
        self.max_length = 37
        self.alphabet_pos = alphabet_pos
        self.alphabet_chunk = alphabet_chunk
        self.alphabet_tag = None
        self.r = Regex()

    def read_conll_format(self, input_file):
        """
        Read data from file including words, chunks, part-of-speech and tagging
        :param input_file: Directory of data file
        :return: a tuple including information about words, chunks, POS, tags, number of sentences and max length
        of a sentence
        """
        with codecs.open(input_file, 'r', 'utf-8') as f:
            word_list = []  # list of sentences that contain words
            chunk_list = [] # list of sentences' chunks
            pos_list = []   # list of sentences' pos
            tag_list = []   # list of sentences' tags
            words = []
            chunks = []
            poss = []
            tags = []
            num_sent = 0
            max_length = 0
            for i, line in enumerate(f):
                line = line.split()
                if len(line) > 0:
                    words.append(self.r.map_word_label(word=line[0].lower()))
                    try:
                        poss.append(line[1])
                        chunks.append(line[2])
                        tags.append(line[3])
                    except:
                        print(i)
                else:
                    word_list.append(words)
                    pos_list.append(poss)
                    chunk_list.append(chunks)
                    tag_list.append(tags)
                    sent_length = len(words)
                    words = []
                    chunks = []
                    poss = []
                    tags = []
                    num_sent += 1
                    max_length = max(max_length, sent_length)
        return word_list, pos_list, chunk_list, tag_list, num_sent, max_length

    def map_string_2_id_open(self, string_list, name):
        """
        Get id of each word in a string list that contains all sentences
        :param string_list: sentences that contain words
        :param name: kind of string
        :return: id of each word in sentences
        """
        string_id_list = []
        alphabet_string = Alphabet(name)
        for strings in string_list:
            ids = []
            for string in strings:
                id = alphabet_string.get_index(string)
                ids.append(id)
            string_id_list.append(ids)
        alphabet_string.close()
        return string_id_list, alphabet_string

    def map_string_2_id_close(self, string_list, alphabet_string):
        string_id_list = []
        for strings in string_list:
            ids = []
            for string in strings:
                id = alphabet_string.get_index(string)
                ids.append(id)
            string_id_list.append(ids)
        return string_id_list

    def map_string_2_id(self, pos_list_train, pos_list_val, pos_list_test, chunk_list_train, chunk_list_val,
                        chunk_list_test, tag_list_train, tag_list_val, tag_list_test):
        pos_id_list_train, alphabet_pos = self.map_string_2_id_open(pos_list_train, 'pos')
        pos_id_list_val = self.map_string_2_id_close(pos_list_val, alphabet_pos)
        pos_id_list_test = self.map_string_2_id_close(pos_list_test, alphabet_pos)

        chunk_id_list_train, alphabet_chunk = self.map_string_2_id_open(chunk_list_train, 'chunk')
        chunk_id_list_val = self.map_string_2_id_close(chunk_list_val, alphabet_chunk)
        chunk_id_list_test = self.map_string_2_id_close(chunk_list_test, alphabet_chunk)

        tag_id_list_train, alphabet_tag = self.map_string_2_id_open(tag_list_train, 'tag')
        tag_id_list_val = self.map_string_2_id_close(tag_list_val, alphabet_tag)
        tag_id_list_test = self.map_string_2_id_close(tag_list_test, alphabet_tag)

        return pos_id_list_train, pos_id_list_val, pos_id_list_test, chunk_id_list_train, chunk_id_list_val, \
               chunk_id_list_test, tag_id_list_train, tag_id_list_val, tag_id_list_test, alphabet_pos, \
               alphabet_chunk, alphabet_tag

    def construct_tensor_word(self, word_sentences, unknown_embed, embed_words, embed_vectors, embed_dim, max_length):
        """
        Choose embedding vector for each word
        :param word_sentences: all sentences that contain words
        :param unknown_embed: values returned when an unknown word comes
        :param embed_words: vocabulary
        :param embed_vectors: embedded vectors corresponding to each word
        :param embed_dim: dimension of embedded vectors
        :param max_length: max number of words in a sentence
        :return: Embedded vectors for all words following order in vocab
        """
        X = np.empty([len(word_sentences), max_length, embed_dim])
        for i in range(len(word_sentences)):
            words = word_sentences[i]
            length = len(words)
            for j in range(length):
                word = words[j].lower()
                try:
                    embed = embed_vectors[embed_words.index(word)]
                except:
                    embed = unknown_embed
                X[i, j] = embed
            # Zero out X after the end of the sequence <=> ZERO_PADDING
            X[i, length:] = np.zeros([1, embed_dim])
        return X

    def construct_tensor_onehot(self, feature_sentences, max_length, dim):
        X = np.zeros([len(feature_sentences), max_length, dim])
        for i in range(len(feature_sentences)):
            for j in range(len(feature_sentences[i])):
                if feature_sentences[i][j] > 0:
                    X[i, j, feature_sentences[i][j]] = 1
        return X

    def create_vector_data(self, 
                           word_list,
                           pos_id_list,
                           chunk_id_list,
                           tag_id_list):

        word_tensor = self.construct_tensor_word(word_list, self.unknown_embedd,
                                                 self.embedd_words, self.embedd_vectors,
                                                 self.embedd_dim, self.max_length)

        # categorical pos tag
        dim_pos = self.alphabet_pos.size()
        pos_tensor = self.construct_tensor_onehot(pos_id_list, self.max_length, dim_pos)

        # categorical chunk tag
        dim_chunk = self.alphabet_chunk.size()
        chunk_tensor = self.construct_tensor_onehot(chunk_id_list, self.max_length, dim_chunk)

        # categorical ner tag
        dim_tag = self.alphabet_tag.size()
        tag_tensor = self.construct_tensor_onehot(tag_id_list, self.max_length, dim_tag)

        return word_tensor, pos_tensor, chunk_tensor, tag_tensor

    def create_vector_data_ex(self,
                              word_list,
                              pos_id_list,
                              chunk_id_list):
        word_tensor = self.construct_tensor_word(word_list, self.unknown_embedd,
                                                 self.embedd_words, self.embedd_vectors,
                                                 self.embedd_dim, self.max_length)

        # categorical pos tag
        dim_pos = self.alphabet_pos.size()
        pos_tensor = self.construct_tensor_onehot(pos_id_list, self.max_length, dim_pos)

        # categorical chunk tag
        dim_chunk = self.alphabet_chunk.size()
        chunk_tensor = self.construct_tensor_onehot(chunk_id_list, self.max_length, dim_chunk)
        feature_vectors = np.concatenate((word_tensor, pos_tensor, chunk_tensor), axis=2)

        return feature_vectors

    def create_data(self, train_dir, val_dir, test_dir):

        # read data
        print("start reading data ...")
        word_list_train, pos_list_train, chunk_list_train, tag_list_train, num_sent_train, max_length_train = \
            self.read_conll_format(train_dir)
        print("done training data...")
        word_list_val, pos_list_val, chunk_list_val, tag_list_val, num_sent_val, max_length_val = \
            self.read_conll_format(val_dir)
        print("done valid data...")
        word_list_test, pos_list_test, chunk_list_test, tag_list_test, num_sent_test, max_length_test = \
            self.read_conll_format(test_dir)
        print("done testing data")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # convert data to id
        print("start converting data to id ....")
        pos_id_list_train, pos_id_list_val, pos_id_list_test, chunk_id_list_train, chunk_id_list_val, \
        chunk_id_list_test, tag_id_list_train, tag_id_list_val, tag_id_list_test, alphabet_pos, \
        alphabet_chunk, alphabet_tag = \
            self.map_string_2_id(pos_list_train, pos_list_val, pos_list_test,
                                 chunk_list_train, chunk_list_val, chunk_list_test,
                                 tag_list_train, tag_list_val, tag_list_test)

        self.alphabet_pos = alphabet_pos
        self.alphabet_tag = alphabet_tag
        self.alphabet_chunk = alphabet_chunk

        # get max_length from train, validattion and test set
        self.max_length = max(max_length_train, max_length_val, max_length_test)

        print("%%%%%%%%%%%%%%%%%%")
        print("Done preparing data")

        train_data = (word_list_train, pos_id_list_train, chunk_id_list_train, tag_id_list_train)

        valid_data = (word_list_val, pos_id_list_val, chunk_id_list_val, tag_id_list_val)

        test_data = (word_list_test, pos_id_list_test, chunk_id_list_test, tag_id_list_test)

        self.save_data()

        return train_data, valid_data, test_data

    def save_data(self, data_model_path=join(dirname(__file__), "model/data")):

        alphabet_pos_path = os.path.join(data_model_path, "pos_data.pkl")
        alphabet_chunk_path = os.path.join(data_model_path, "chunk_data.pkl")
        alphabet_tag_path = os.path.join(data_model_path, "tag_data.pkl")

        with open(alphabet_pos_path, 'wb') as fp:
            pkl.dump(self.alphabet_pos, fp)
            fp.close()

        with open(alphabet_chunk_path, 'wb') as fp:
            pkl.dump(self.alphabet_chunk, fp)
            fp.close()

        with open(alphabet_tag_path, 'wb') as fp:
            pkl.dump(self.alphabet_tag, fp)
            fp.close()

    def check_data_save(self, data_model_path=join(dirname(__file__), "model/data")):
        if os.path.isdir(data_model_path) and os.listdir(data_model_path) == 3:
            return True
        else:
            return False

    def load_data(self, data_model_path, type_):

        alphabet_pos_path = os.path.join(data_model_path, "pos_data.pkl")
        alphabet_chunk_path = os.path.join(data_model_path, "chunk_data.pkl")
        alphabet_tag_path = os.path.join(data_model_path, "tag_data.pkl")

        if type_ == 'pos_data':
            with open(alphabet_pos_path, 'rb') as fp:
                pos_data = pkl.load(fp)
                fp.close()
            return pos_data

        elif type_ == 'chunk_data':
            with open(alphabet_chunk_path, 'rb') as fp:
                chunk_data = pkl.load(fp)
                fp.close()
            return chunk_data

        elif type_ == 'tag_data':
            with open(alphabet_tag_path, 'rb') as fp:
                tag_data = pkl.load(fp)
                fp.close()
            return tag_data

        else:
            return None

    def predict_to_file(self, predicts, tests, alphabet_tag, output_file):
        print(alphabet_tag.instance2index)
        with codecs.open(output_file, 'w', 'utf-8') as f:
            for i in range(len(tests)):
                for j in range(len(tests[i])):
                    predict = alphabet_tag.get_instance(predicts[i][j])
                    if predict == None:
                        predict = alphabet_tag.get_instance(predicts[i][j] + 1)

                    test = alphabet_tag.get_instance(tests[i][j])
                    f.write('_' + ' ' + predict + ' ' + test + '\n')
                f.write('\n')

    def mkdir(self, dir):
        if (os.path.exists(dir) == False):
            os.mkdir(dir)