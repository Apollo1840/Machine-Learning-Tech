from semi_supervised_clustering_gmm import *
from semi_supervised_clustering_kmean import *
from semi_supervised_clustering_label_propagation import *
from semi_supervised_clustering_self_training import *


def plot_multi_method_accs(methods_accs, percentages, method_names):
    """
    Plot multiple methods' accuracies on a single chart.

    :param methods_accs: A list of lists containing accuracies for each method.
    :param percentages: The percentages of labeled data used.
    :param method_names: The names of the methods to be plotted.
    """
    percentage_labels = [f"{p * 100}%" for p in percentages]

    plt.figure(figsize=(10, 6))
    for method_acc, name in zip(methods_accs, method_names):
        # Calculate the mean accuracy for each percentage
        mean_accs = np.array([sum(acc_round) / len(acc_round) for acc_round in method_acc])
        std_accs = np.array([np.std(acc_round) for acc_round in method_acc])
        plt.plot(percentage_labels, mean_accs, label=name)
        plt.fill_between(percentage_labels, mean_accs - std_accs, mean_accs + std_accs, alpha=0.2)

    plt.xlabel('Percentage of Labeled Data')
    plt.ylabel('Clustering Accuracy')
    plt.title('Comparison of Semi-Supervised Clustering Methods')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Prepare the MNIST data
    x_train, y_train, x_test, y_test = load_data()
    percentages = [0.02, 0.05, 0.1, 0.2, 0.5]
    rounds = 5

    # Initialize the main list to store accuracies of all methods
    methods_accs = []

    # List of methods
    methods = [SemiSupervisedGMM, CopKMean, LabelPropagation, SVC]

    for Method in tqdm(methods, desc="methods"):
        accs = []  # Temporarily store accuracies for the current method
        for percentage in tqdm(percentages, desc=">> percentages"):
            accs_round = []
            for _ in tqdm(range(rounds), desc="rounds"):
                # Prepare labeled and unlabeled data
                labeled_data, labeled_labels, unlabeled_data, unlabeled_labels = prepare_labeled_data(
                    x_train, y_train, percentage=percentage)
                X = np.concatenate((labeled_data, unlabeled_data), axis=0)
                label_assignments = list(labeled_labels) + list(unlabeled_labels)

                if Method is SemiSupervisedGMM:
                    model = Method(n_components=10)
                    model.fit(X, labeled_data, labeled_labels)
                elif Method is CopKMean:
                    model = Method(n_components=10)
                    model.fit(X, labeled_labels)
                elif Method is LabelPropagation:
                    model = Method()
                    model.fit(X, np.concatenate((labeled_labels, [-1] * len(unlabeled_labels))))
                else:  # SVM
                    model = Method()
                    model.fit(labeled_data, labeled_labels)

                # Predict cluster assignments
                cluster_assignments = model.predict(X)

                # Calculate and append accuracy
                accs_round.append(clustering_accuracy(label_assignments, cluster_assignments))

            accs.append(accs_round)

        # Append the accuracies of the current method to the main list
        methods_accs.append(accs)

    method_names = ['SemiSupervisedGMM', 'CopKMean', 'LabelPropagation', 'SVC']
    plot_multi_method_accs(methods_accs, percentages, method_names)
