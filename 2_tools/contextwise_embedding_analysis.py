from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


def mnist_data(as_img=False):
    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    # X shape (60,000 28x28), y shape (10,000, )
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    x_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    x_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    if as_img:
        x_train = x_train.reshape((len(x_train), 28, 28, 1))
        x_test = x_test.reshape((len(x_test), 28, 28, 1))

    return x_train, y_train, x_test, y_test


def find_maxset_similiar(items,
                         func_blur=lambda item: item,
                         func_eval=lambda item: item,
                         func_valid=lambda items: True):
    """
    todo: test this function
    
    return the maxset with repeating the blurring operation.

    :param items, List
    :param func_blur: function handler, Item -> Item
        few rounds of blurring must make sure func_valid can definitly pass.
    :param func_eval: functional handler , func_eval(item) = something which supports hashing equality
    :param func_valid: function handler, List -> Bool
    :return List: List of items
    """

    items_org = items

    maxset, indices_maxset = find_maxset(items, func_eval=func_eval)
    i = 0
    maximum_trials = 1000
    while not func_valid(maxset) and i <= maximum_trials:
        items = [func_blur(item) for item in items]
        maxset, indices_maxset = find_maxset(items, func_eval=func_eval)
        i += 1
    items_maxset = [items_org[i] for i in indices_maxset]
    return items_maxset


def find_maxset(items, func_eval=lambda x: x, return_indices=True):
    """

    > find_maxset([1.2, 2, 1.05, 3.1, 1.1], func_eval=lambda x: int(x))
    > # [1.2, 1.05, 1.1]

    :param return_indices: bool
    :param func_eval: functional handler , func_eval(item) = something which supports hashing equality
    :param items: List
    :return:
    """

    values = [func_eval(item) for item in items]
    indices_maxset = locate_maxset(values)
    maxset = [items[i] for i in indices_maxset]

    if return_indices:
        return maxset, indices_maxset
    else:
        return maxset


def locate_maxset(items):
    """

    > locate_maxset([1, 2, 1, 3, 1])
    > # [0, 2, 4]

    :param items: List
    :return: List[int], location of the maximum appearance item
    """

    sort_dict = sorting_items(items)
    count_tuple = []
    for item, indices in sort_dict.items():
        count_tuple.append((item, len(indices)))
    count_tuple = sorted(count_tuple, key=lambda x: x[1])

    item_max = count_tuple[-1][0]
    indices_maxset = sort_dict[item_max]
    return indices_maxset


def sorting_items(items):
    """

    > sorting_items([1, 2, 1, 3, 1])
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


def blur_ta_subsampling(x_sig, level=2):
    x_sig = resample(x_sig, len(x_sig) // level)
    levels = len(x_sig) // level
    x_sig = pd.cut(x_sig, levels, labels=[str(i) for i in range(levels)])
    x_sig_v = [int(xi) for xi in x_sig]
    return x_sig_v


def blur_ta_subsampling_normalize(x_sig, level=2):
    """
    time and amplitude subsampling

    """
    x_sig_v = blur_ta_subsampling(x_sig, level)

    # normalize
    max_ = max(x_sig_v)
    x_sig_v = [xi / max_ for xi in x_sig_v]
    return x_sig_v


def eval_list2str(list):
    values = [round(v, 2) for v in list]
    return "_".join(values)


def valid_four(list):
    return len(list) >= 4


def show_effect(x):
    plot_pic_sig(x, 28)

    x = blur_ta_subsampling_normalize(x, 2)
    plt.imshow(np.reshape(x, (28, 28 // 2)))
    plt.show()
    plt.plot(x)
    plt.show()

    x = blur_ta_subsampling_normalize(x, 2)
    plt.imshow(np.reshape(x, (28, 28 // 4)))
    plt.show()
    plt.plot(x)
    plt.show()


def plot_pic_sig(x, size):
    plt.imshow(np.reshape(x, (28, size)))
    plt.show()
    plt.plot(x)
    plt.show()


# get 4 similar tokens under different context(from 2 labels), show the embedding fitness.

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mnist_data()
    print(x_train.shape)

    x = x_train[0]
    # show_effect(x)

    maxset = find_max_similiar(
        x_train,
        func_blur=lambda item: blur_ta_subsampling_normalize(item),
        func_eval=eval_list2str,
        func_valid=valid_four
    )

    for x in maxset:
        plt.imshow(np.reshape(x, (28, 28)))
        plt.show()
