from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


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

    return x_train, y_train, x_test, y_test


def find_maxset(items, func_eval=lambda x: x):
    """

    > find_maxset([1, 2, 1, 3, 1])
    > #

    :param func_eval: functional handler , func_eval(item) = something which supports hashing equality
    :param items: List
    :return:
    """

    values = [func_eval(item) for item in items]
    indices_maxset = locate_maxset(values)
    maxset = [items[i] for i in indices_maxset]
    return maxset


def locate_maxset(items):
    """

    > locate_maxset([1, 2, 1, 3, 1])
    > # [0, 2, 4]

    :param items: List
    :return: List[int], location of the maximum appearance item
    """

    sort_dict = sorting(items)
    count_tuple = []
    for item, indices in sort_dict.items():
        count_tuple.append((item, len(indices)))
    count_tuple = sorted(count_tuple, key=lambda x: x[1])

    item_max = count_tuple[-1][0]
    indices_maxset = sort_dict[item_max]
    return indices_maxset


def sorting(items):
    """

    > sorting([1, 2, 1, 3, 1])
    > # {1: [0,2,4], 2: [1], 3:[3] }

    :param items: List
    :return: Dict: (item: indices)
    """
    sort_dict = {}
    for i, item in enumerate(items):
        if item in sort_dict:
            sort_dict[item].append(i)
        else:
            sort_dict[item] = [i]
    return sort_dict


def plot_pic_sig(x, size):
    plt.imshow(np.reshape(x, (size, size)))
    plt.show()
    plt.plot(x)
    plt.show()


def compress_subsample(x_sig, level=2):
    x_sig = resample(x_sig, len(x_sig) // level)
    levels = len(x_sig) // level
    x_sig = pd.cut(x_sig, levels, labels=[str(i) for i in range(levels)])
    x_sig_v = [int(xi) for xi in x_sig]
    return x_sig_v


def compress_subsample_normal(x_sig, level=2):
    x_sig_v = compress_subsample(x_sig, level)

    # normalize
    max_ = max(x_sig_v)
    x_sig_v = [xi / max_ for xi in x_sig_v]
    return x_sig_v


def show_effect(x):
    plot_pic_sig(x, 28)

    x = compress_subsample_normal(x, 2)
    plt.imshow(np.reshape(x, (28, 28 // 2)))
    plt.show()
    plt.plot(x)
    plt.show()

    x = compress_subsample_normal(x, 2)
    plt.imshow(np.reshape(x, (28, 28 // 4)))
    plt.show()
    plt.plot(x)
    plt.show()


# get 4 similar tokens under different context(from 2 labels), show the embedding fitness.

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mnist_data()
    print(x_train.shape)

    x = x_train[0]
    # show_effect(x)

    x = [compress_subsample(xi) for xi in x]
