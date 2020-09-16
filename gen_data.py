import numpy as np

from Utils import Utils
from tensorflow import keras

ID_WORD = 0
ID_POS = 1
ID_CHUNK = 2
ID_TAG = 3


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, utils: Utils, batch_size=64, shuffle=True):
        'Initialization'
        self.utils = utils
        self.word_list = np.array(data[ID_WORD])
        self.pos_id_list = np.array(data[ID_POS])
        self.chunk_id_list = np.array(data[ID_CHUNK])
        self.tag_id_list = np.array(data[ID_TAG])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.word_list.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.word_list.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        words = self.word_list[indexes]
        pos_ids = self.pos_id_list[indexes]
        chunk_ids = self.chunk_id_list[indexes]
        tag_ids = self.tag_id_list[indexes]

        word_tensor, pos_tensor, chunk_tensor, tag_tensor = self.utils.create_vector_data(words,
                                                                                          pos_ids,
                                                                                          chunk_ids,
                                                                                          tag_ids)
        X = np.concatenate((word_tensor, pos_tensor, chunk_tensor), axis=2)
        y = tag_tensor
        # Generate data
        # for i, ID in enumerate(indexes):
        #     embed_word = self.embedded_words[ID]
        #     pos_id = self.post_ids[ID]
        #     chunk_id = self.chunk_ids[ID]
        #
        #     # concat
        #     input_train = embed_word
        #     input_train = np.concatenate((input_train, pos_id), axis=1)
        #     input_train = np.concatenate((input_train, chunk_id), axis=1)
        #     X[i,] = input_train
        #
        #     # Store class
        #     y[i] = self.labels[ID]

        return X, y
