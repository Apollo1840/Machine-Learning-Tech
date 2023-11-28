import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from cop_k_mean import *


def mnist_data(reshape=False, categorical=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if reshape:
        # Reshape data
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    else:
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

    if not categorical:
        # Convert class vectors to binary class matrices
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def prepare_labeled_data(x_train, y_train, percentage, sort=True):
    """
    Prepare subsets of data with varying percentages of labeled data.

    Parameters:
    x_train (numpy.ndarray): Training data features.
    y_train (numpy.ndarray): Training data labels.
    percentages (list): List of percentages of labeled data.
    sort (bool, optional): Whether to sort labeled_labels along with labeled_data. Default is True.

    Returns:
    dict: A dictionary where keys are percentages and values are tuples of
          (labeled_data, labeled_labels, unlabeled_data).
    """
    n_samples = x_train.shape[0]

    n_labeled = int(n_samples * percentage)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]

    labeled_data = x_train[labeled_indices, :]
    labeled_labels = y_train[labeled_indices]

    if sort:
        # Sort labeled_labels along with labeled_data
        sorted_indices = np.argsort(labeled_labels)
        labeled_data = labeled_data[sorted_indices, :]
        labeled_labels = labeled_labels[sorted_indices]

    unlabeled_data = x_train[unlabeled_indices, :]
    unlabeled_labels = y_train[unlabeled_indices]  # for analysis/visualization

    return labeled_data, labeled_labels, unlabeled_data, unlabeled_labels


if __name__ == "__main__":
    # prepare data
    x_train, y_train, x_test, y_test = mnist_data(categorical=True)
    # percentages = [0.01, 0.05, 0.1, 0.2, 0.5]
    labeled_data, labeled_labels, unlabeled_data, unlabeled_labels = prepare_labeled_data(x_train, y_train, percentage=0.01)
    X = np.concatenate((labeled_data, unlabeled_data), axis=0)

    print(X.shape)
    print(len(labeled_data), len(labeled_labels))
    print(set(labeled_labels), set(y_train))

    # clustering
    Method = CopKMean
    must_link, cannot_link = Method.label_to_constraints(labeled_labels)
    # must_link, cannot_link = Method.label_to_constraints(labeled_labels, len(X))
    cluster_assignments, centroids = Method.fit_transform(X, 10, must_link, cannot_link, verbose=True)

    # label_assignments = list(labeled_labels) + list(unlabeled_labels)
    # Method.scatter_cluster_points_with_labeled(X, label_assignments, labeled_labels)
    Method.scatter_cluster_points_with_labeled(X, cluster_assignments, labeled_labels)
