import numpy as np

from Utils import Utils
from tensorflow import keras


ID_WORD_EMBEDDING = 0
ID_POST_ID = 1
ID_CHUNK_ID = 2
ID_TAG_ID = 3


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=64, shuffle=True):
        'Initialization'
        self.embedded_words = data[ID_WORD_EMBEDDING]
        self.post_ids = data[ID_POST_ID]
        self.chunk_ids = data[ID_CHUNK_ID]
        self.labels = data[ID_TAG_ID]
        self.dim = (data[ID_WORD_EMBEDDING].shape[1],
                    data[ID_WORD_EMBEDDING].shape[2] + data[ID_POST_ID].shape[2] + data[ID_CHUNK_ID].shape[2])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.embedded_words.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.embedded_words.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.labels.shape[1:]), dtype=int)

        # Generate data
        for i, ID in enumerate(indexes):
            embed_word = self.embedded_words[ID]
            pos_id = self.post_ids[ID]
            chunk_id = self.chunk_ids[ID]

            # concat
            input_train = embed_word
            input_train = np.concatenate((input_train, pos_id), axis=1)
            input_train = np.concatenate((input_train, chunk_id), axis=1)
            X[i,] = input_train

            # Store class
            y[i] = self.labels[ID]

        return X, y