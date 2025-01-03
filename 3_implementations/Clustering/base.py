import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

from scipy.optimize import linear_sum_assignment


class SemiSupervisedClustering():
    # taken from scikit-learn (https://goo.gl/1RYPP5)

    def __init__(self, n_components, max_iter=300, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    @classmethod
    def initialize_centroids(cls, X, k, method="random", verbose=False):
        if method == 'random':
            ids = list(range(len(X)))
            random.shuffle(ids)
            return [X[i] for i in ids[:k]]

        elif method == 'kmpp':
            X_np = np.array(X)
            n_samples, n_features = X_np.shape

            # Initialize the first center randomly
            centers = [X_np[random.randint(0, n_samples - 1)]]

            # Initialize an array to store distances
            closest_dist_sq = np.full(n_samples, np.inf)

            # Iterate over the remaining centers
            for _ in tqdm(range(1, k), desc="initialize centroids", total=k - 1, disabel=not verbose):
                # Update the distance array for the newest center only
                dist_sq = np.sum((X_np - centers[-1]) ** 2, axis=1)
                closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

                # Choose the next center with a probability proportional to the square of the distance
                probabilities = closest_dist_sq / closest_dist_sq.sum()
                cumulative_probabilities = np.cumsum(probabilities)
                r = random.random()
                next_center_index = np.searchsorted(cumulative_probabilities, r)
                centers.append(X_np[next_center_index])

            return centers

    @classmethod
    def calculate_tolerance(cls, tol, X):
        X_np = np.array(X)
        variances = np.var(X_np, axis=0)
        return tol * np.mean(variances)

    @classmethod
    def update_centroids(cls, X, assignments, k):
        """ Update centroids based on current cluster assignments """
        n_features = X.shape[1]
        centroids = np.zeros((k, n_features))

        for cluster_idx in range(k):
            cluster_points = X[assignments == cluster_idx]
            if len(cluster_points) > 0:
                centroids[cluster_idx] = np.mean(cluster_points, axis=0)

        return centroids

    @classmethod
    def scatter_cluster_points_with_labeled(cls, X, label_assignments, cluster_assignments, n_labeled, n_samples=6000,
                                            split=False):
        # Create a t-SNE model
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

        # Fit the model to your data (X)
        X_tsne = tsne.fit_transform(X[:n_samples])

        cluster_assignments = cls.rematch_cluster_assignments(label_assignments, cluster_assignments)

        # Scatter plot for all data points with cluster assignments
        plt.figure(figsize=(10, 6))

        cmap = plt.get_cmap('rainbow', len(set(label_assignments)))

        plt.scatter(X_tsne[n_labeled:n_samples, 0], X_tsne[n_labeled:n_samples, 1],
                    c=cluster_assignments[n_labeled:n_samples], s=100, cmap=cmap, alpha=0.15)
        plt.colorbar(label='Cluster Assignments')

        if split:
            plt.title('t-SNE Visualization with Cluster Assignments')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.show()
            plt.figure(figsize=(10, 6))

        plt.scatter(X_tsne[:n_labeled, 0], X_tsne[:n_labeled, 1], s=20, c=label_assignments[:n_labeled],
                    cmap=cmap, edgecolors="black", linewidth=1, alpha=1)
        plt.colorbar(label='Label Assignments')

        plt.title('t-SNE Visualization with Cluster Assignments')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    @classmethod
    def labels_to_constraints(cls, label_indices, verbose=False):
        """

        :param label_indices: eg. [0, 0, 0, 1, 1, 2, 2]
        :return: must_link, cannot_link:
             - must_link : ({0: {1, 2}, 1: {0, 2}, 2: {0, 1}, 3: {4}, 4: {3}, 5: {6}, 6: {5}},
             - cannot_link : {
                        0: {3, 4, 5, 6},
                        1: {3, 4, 5, 6},
                        2: {3, 4, 5, 6},
                        3: {0, 1, 2, 5, 6},
                        4: {0, 1, 2, 5, 6},
                        5: {0, 1, 2, 3, 4},
                        6: {0, 1, 2, 3, 4}})
        """
        n = len(label_indices)
        must_link, cannot_link = dict(), dict()

        # create a label_indices_map: {label: (start_index, end_index)}
        label_indices_map = dict()

        start_index = 0
        current_label = label_indices[start_index]
        for i in range(n):
            if label_indices[i] != current_label:
                label_indices_map.update({current_label: (start_index, i)})
                start_index = i
                current_label = label_indices[start_index]
        label_indices_map.update({current_label: (start_index, n)})

        for i in tqdm(range(n), desc="label to constraints", disable=not verbose):
            start, end = label_indices_map[label_indices[i]]
            must_link.update({i: {j for j in range(start, end) if i != j}})
            cannot_link.update({i: {j for j in range(start)} | {j for j in range(end, n)}})
        return must_link, cannot_link

    @staticmethod
    def tuples_to_constraints(must_link, cannot_link, n):
        """
        Example:
                  - must_link = [(0, 1), (3, 4)]
                  - cannot_link = [(0, 2)]


        :param must_link:
        :param cannot_link:
        :param n: number of points
        :return:
        """
        must_link = list() if must_link is None else must_link
        cannot_link = list() if cannot_link is None else cannot_link

        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in must_link:
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in cannot_link:
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)

        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('Inconsistent constraints between %d and %d' % (i, j))

        return ml_graph, cl_graph

    @staticmethod
    def euclidean_distance(x1, x2):
        return euclidean_distances([x1], [x2])[0][0]

    @staticmethod
    def rematch_cluster_assignments(label_assignments, cluster_assignments):
        """
        Rematch cluster assignments based on the Hungarian algorithm to align with label assignments.

        :param label_assignments: List of true labels.
        :param cluster_assignments: List of assigned cluster labels.
        :return: Rematched cluster assignments.
        """
        # Create a confusion matrix
        n_clusters = max(max(label_assignments), max(cluster_assignments)) + 1
        confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)

        for true_label, cluster_label in zip(label_assignments, cluster_assignments):
            confusion_matrix[true_label][cluster_label] += 1

        # Apply the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Map from old to new cluster assignments
        remap = {old: new for old, new in zip(col_ind, row_ind)}

        # Rematch cluster assignments
        rematched_cluster_assignments = [remap[cluster] for cluster in cluster_assignments]

        return rematched_cluster_assignments
