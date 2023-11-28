import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances


class SemiSupervisedClustering():
    # taken from scikit-learn (https://goo.gl/1RYPP5)

    @classmethod
    def initialize_centroids(cls, X, k, method="random"):
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
            for _ in tqdm(range(1, k), desc="initialize centroids", total=k-1):
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

    @staticmethod
    def scatter_cluster_points_with_labeled(X, cluster_assignments, labeled_labels, n_samples=6000):
        # Create a t-SNE model
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

        # Fit the model to your data (X)
        X_tsne = tsne.fit_transform(X[:n_samples])

        # Scatter plot for all data points with cluster assignments
        plt.figure(figsize=(10, 6))

        plt.scatter(X_tsne[:len(labeled_labels), 0], X_tsne[:len(labeled_labels), 1], s=20, c=labeled_labels,
                    cmap='rainbow', marker="x", alpha=1)
        plt.colorbar(label='Label Assignments')

        plt.scatter(X_tsne[len(labeled_labels):6000, 0], X_tsne[len(labeled_labels):6000, 1],
                    c=cluster_assignments[len(labeled_labels):6000], s=100, cmap='rainbow', alpha=0.05)
        plt.colorbar(label='Cluster Assignments')

        plt.title('t-SNE Visualization with Cluster Assignments')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    @classmethod
    def label_to_constraints(cls, label_indices):
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
        for i in tqdm(range(n)):
            if label_indices[i] != current_label:
                label_indices_map.update({current_label: (start_index, i)})
                start_index = i
                current_label = label_indices[start_index]
        label_indices_map.update({current_label: (start_index, n)})

        for i in tqdm(range(n), desc="label to constraints"):
            start, end = label_indices_map[label_indices[i]]
            must_link.update({i: {j for j in range(start, end) if i != j}})
            cannot_link.update({i: {j for j in range(start)} | {j for j in range(end, n)}})
        return must_link, cannot_link

    @staticmethod
    def convert_and_expand_constraints(must_link, cannot_link, n):
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
