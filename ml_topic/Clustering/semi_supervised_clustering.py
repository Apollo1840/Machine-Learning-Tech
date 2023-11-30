import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from cop_k_mean import *


def load_data(dataset="mnist", reshape=False, categorical=False, n_samples=6000):
    """


    :param dataset:
    :param reshape: True: for CNN; False: for MLP
    :param categorical: True: as onehot; False: numerical
    :param n_samples:
    :return:
    """
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train[10:, :]
        y_train = y_train[10:, 0]
        y_test = y_test[:, 0]
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if reshape:
        # If reshape is True, data is already in the correct shape for CNNs
        pass
    else:
        # Flatten the data for fully connected networks
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

    if categorical:
        # Convert class vectors to binary class matrices (one-hot encoding)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    if n_samples is not None and n_samples < x_train.shape[0]:
        # Randomly sample n_samples from the datasets
        train_indices = np.random.choice(x_train.shape[0], n_samples, replace=False)
        x_train = x_train[train_indices, :]
        y_train = y_train[train_indices]

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


def clustering_accuracy(label_assignments, cluster_assignments):
    """
    Calculate the clustering accuracy using rematched cluster assignments.

    :param label_assignments: List of true labels.
    :param cluster_assignments: List of assigned cluster labels.
    :return: Clustering accuracy.
    """
    # Rematch cluster assignments
    rematched_clusters = SemiSupervisedClustering.rematch_cluster_assignments(label_assignments, cluster_assignments)

    # Calculate accuracy
    correct_matches = sum(l == c for l, c in zip(label_assignments, rematched_clusters))
    total_elements = len(label_assignments)
    accuracy = correct_matches / total_elements

    return accuracy


def plot_clustering_accs(accs, percentages):
    # Convert percentages to string for plotting
    percentage_labels = [f"{p * 100}%" for p in percentages]

    # Calculate mean and standard deviation for each percentage
    mean_accs = np.mean(accs, axis=1)
    std_devs = np.std(accs, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(percentage_labels, mean_accs, linestyle='-', marker='o', label='Mean Accuracy')
    plt.fill_between(percentage_labels, mean_accs - std_devs, mean_accs + std_devs, alpha=0.2)
    plt.title('Clustering Accuracy vs Label Percentage (with Confidence Interval)')
    plt.xlabel('Label Percentage')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # prepare data
    x_train, y_train, x_test, y_test = load_data()
    Method = CopKMean
    rounds = 5

    accs = []
    percentages = [0.01, 0.05, 0.1, 0.2, 0.5]
    for percentage in tqdm(percentages, desc=">> percentages"):
        accs_round = []
        for _ in range(rounds):
            labeled_data, labeled_labels, unlabeled_data, unlabeled_labels = prepare_labeled_data(x_train, y_train,
                                                                                                  percentage=percentage)
            X = np.concatenate((labeled_data, unlabeled_data), axis=0)
            label_assignments = list(labeled_labels) + list(unlabeled_labels)

            # print(X.shape)
            # print(len(labeled_data), len(labeled_labels))
            # print(set(labeled_labels), set(y_train))

            # clustering
            must_link, cannot_link = Method.labels_to_constraints(labeled_labels, verbose=False)
            cluster_assignments, centroids = Method.fit_transform(X, 10, must_link, cannot_link, verbose=False)

            # Method.scatter_cluster_points_with_labeled(X, label_assignments, cluster_assignments, n_labeled=len(labeled_labels))
            accs_round.append(clustering_accuracy(label_assignments, cluster_assignments))
        accs.append(accs_round)

    plot_clustering_accs(accs, percentages)
