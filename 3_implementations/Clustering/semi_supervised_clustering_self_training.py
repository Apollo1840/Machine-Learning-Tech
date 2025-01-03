from sklearn.svm import SVC
from semi_supervised_clustering import *


# Example usage
if __name__ == "__main__":
    # prepare data
    x_train, y_train, x_test, y_test = load_data()
    Method = SVC
    # Method = LabelSpreading
    rounds = 5

    accs = []
    percentages = [0.01, 0.05, 0.1, 0.2, 0.5]
    for percentage in tqdm(percentages, desc=">> percentages"):
        accs_round = []
        for _ in tqdm(range(rounds), desc="rounds"):
            labeled_data, labeled_labels, unlabeled_data, unlabeled_labels = prepare_labeled_data(x_train, y_train,
                                                                                                  percentage=percentage)
            X = np.concatenate((labeled_data, unlabeled_data), axis=0)
            label_assignments = list(labeled_labels) + list(unlabeled_labels)

            # Preparing labels for semi-supervised learning
            labels = np.concatenate((labeled_labels, [-1] * len(unlabeled_labels)))  # -1 denotes unlabeled points

            # Label Propagation
            svm_model = Method()
            svm_model.fit(labeled_data, labeled_labels)
            cluster_assignments = svm_model.predict(X)
            # centroids = SemiSupervisedClustering.update_centroids(X, cluster_assignments, 10)

            # Method.scatter_cluster_points_with_labeled(X, label_assignments, cluster_assignments, n_labeled=len(labeled_labels))
            accs_round.append(clustering_accuracy(label_assignments, cluster_assignments))
        accs.append(accs_round)

    plot_clustering_accs(accs, percentages)