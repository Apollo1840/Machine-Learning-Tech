import numpy as np
import keras


class BatchGenerator:

    def __init__(self, x, y, batch_size):
        assert len(x) == len(y)

        x, y = self.shuffle_xy(x, y)
        self.x = list(x)
        self.y = list(y)
        self.batch_size = batch_size

    def shuffle_xy(self, x, y):
        """
        shuffle the x as well as y in same order.

        :param x:
        :param y:
        :return:
        """

        pair = list(zip(list(x), list(y)))

        np.random.shuffle(pair)
        return zip(*pair)

    def next_batch(self):
        batch_x = []
        batch_y = []
        for _ in range(self.batch_size):
            batch_x.append(self.x.pop())
            batch_y.append(self.y.pop())
        return batch_x, batch_y


class DataGeneratorLoad(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs,
                 labels,
                 batch_size=32,
                 dim=(32, 32, 32),
                 n_channels=1,
                 n_classes=10,
                 shuffle=True):
        """

        :param list_IDs: list of item_ids, use item_id to load x
        :param labels: list of annotation, labels[ID[i]] is the y of i-th item
        :param batch_size:
        :param dim:
        :param n_channels:
        :param n_classes:
        :param shuffle:
        """

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGeneratorBatchGen(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, Y,
                 preprocess_func=None,
                 batch_size=32,
                 shuffle=True):
        """

        :param X: List of np.array
        :param Y: np.array
        :param preprocess_func: preprocess function for the whole batch.
        :param batch_size: Int
        :param shuffle: Bool
        """

        self.X = X
        self.Y = Y
        self.n_samples = len(X)
        self.indexes = None
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.preprocess_func = preprocess_func
        if not self.preprocess_func:
            print("no preprocessing")

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x_batch = []
        for i in indexes:
            x_batch.append(self.X[i])

        if self.preprocess_func is not None:
            x_batch = self.preprocess_func(x_batch)
        return np.array(x_batch), self.Y[indexes]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
