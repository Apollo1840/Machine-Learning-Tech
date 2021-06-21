import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


def trival_data():
    # data
    train_labels = []
    train_samples = []
    for i in range(1000):
        train_labels.append(0)
        train_samples.append(randint(13, 64))
        train_labels.append(1)
        train_samples.append(randint(65, 100))
    for i in range(50):
        train_labels.append(1)
        train_samples.append(randint(13, 64))
        train_labels.append(0)
        train_samples.append(randint(65, 100))

    y_train = np.array(train_labels)
    x_train = np.array(train_samples)

    # preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train.reshape(-1, 1))

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train


def trival_data2():
    # create some data
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)  # randomize the data
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

    x_train, y_train = X[:160], Y[:160]  # first 160 data points
    x_test, y_test = X[160:], Y[160:]  # last 40 data points

    print(x_train.shape)
    print(x_test.shape)

    return x_train, y_train, x_test, y_test


def mnist_data():
    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    # X shape (60,000 28x28), y shape (10,000, )
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    x_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    x_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    print(x_train.shape)
    print(x_test.shape)

    return x_train, y_train, x_test, y_test


def imdb_data():
    """
    input: movie reviews.
    output: pos or neg
    """
    from keras.datasets import imdb

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(y_test), "Validation sequences")
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, y_train, x_test, y_test, vocab_size, maxlen