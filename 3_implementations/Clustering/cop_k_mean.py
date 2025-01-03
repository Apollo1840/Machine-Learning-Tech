import numpy as np
from tqdm import tqdm

from sklearn.metrics.pairwise import euclidean_distances

from base import SemiSupervisedClustering


class CopKMean(SemiSupervisedClustering):

    def fit(self, X, labels=None, must_link=None, cannot_link=None, initialization='random', verbose=False):
        if labels is not None:
            self.must_link, self.cannot_link = self.labels_to_constraints(labels, verbose=verbose)
        else:
            self.must_link, self.cannot_link = must_link, cannot_link

        self.tol = self.calculate_tolerance(self.tol, X)

        if verbose:
            print("initializing centroids...")

        self.centroids = self.initialize_centroids(X, self.n_components, initialization, verbose=verbose)

    def predict(self, X, verbose=False):
        cluster_assignments = np.zeros(X.shape[0], dtype=int)

        for _ in tqdm(range(self.max_iter), desc="iterations", disable=not verbose):
            cluster_assignments = self.assign_clusters_constrained(X, self.centroids, self.must_link, self.cannot_link,
                                                                   verbose=verbose)

            # Update centroids
            new_centroids = self.update_centroids(X, cluster_assignments, self.n_components)

            # Check for convergence based on centroids shift
            shift = sum(self.euclidean_distance(new_centroids[i], self.centroids[i]) for i in range(self.n_components))
            if verbose:
                print("updated centroids with total distance as: {:.3}/{:.3}".format(shift, self.tol))
            if shift <= self.tol:
                break

            self.centroids = new_centroids

        return cluster_assignments

    @classmethod
    def fit_transform(cls, X, k, must_link=None, cannot_link=None, initialization='random', max_iter=300, tol=1e-4,
                      verbose=False):
        """ Constrained K-Means Algorithm

        Example:
            must_link = {0: {1}, 1: {0}, 3: {4}, 4: {3}}
            cannot_link = {0: {2}}

        """
        must_link = dict() if must_link is None else must_link
        cannot_link = dict() if cannot_link is None else cannot_link

        tol = cls.calculate_tolerance(tol, X)

        if verbose:
            print("initializing centroids...")

        centroids = cls.initialize_centroids(X, k, initialization, verbose=verbose)
        cluster_assignments = np.zeros(X.shape[0], dtype=int)

        for _ in tqdm(range(max_iter), desc="iterations", disable=not verbose):
            cluster_assignments = cls.assign_clusters_constrained(X, centroids, must_link, cannot_link, verbose=verbose)

            # Update centroids
            new_centroids = cls.update_centroids(X, cluster_assignments, k)

            # Check for convergence based on centroids shift
            shift = sum(cls.euclidean_distance(new_centroids[i], centroids[i]) for i in range(k))
            if verbose:
                print("updated centroids with total distance as: {:.3}/{:.3}".format(shift, tol))
            if shift <= tol:
                break

            centroids = new_centroids

        return cluster_assignments, centroids

    @classmethod
    def assign_clusters_constrained(cls, X, centroids, must_link, cannot_link, verbose=False):
        """
        Assign clusters with respect to must-link and cannot-link constraints.

        Parameters:
        X (numpy.ndarray): Data points.
        centroids (numpy.ndarray): Current centroids.
        must_link (dict): Dictionary of must-link constraints.
        cannot_link (dict): Dictionary of cannot-link constraints.

        Returns:
        numpy.ndarray: Cluster assignments.
        """
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=int)

        # Calculate distances between each point and each centroid
        distances = euclidean_distances(X, centroids)

        for i in tqdm(range(n_samples), desc="assign cluster index", disable=not verbose):
            # Find the closest centroid
            closest_centroids = np.argsort(distances[i])

            for centroid_index in closest_centroids:
                can_assign = True

                # Check must-link constraints
                if i in must_link:
                    linked_points = must_link[i]
                    if not all(assignments[j] == centroid_index or assignments[j] == 0 for j in linked_points):
                        can_assign = False

                # Check cannot-link constraints
                if can_assign and i in cannot_link:
                    linked_points = cannot_link[i]
                    if any(assignments[j] == centroid_index for j in linked_points):
                        can_assign = False

                # Assign to the closest valid centroid
                if can_assign:
                    assignments[i] = centroid_index
                    break

        return assignments


class CopKMean2(CopKMean):
    """
    needs less RAM. (in case X is too big)

    """

    def assign_clusters_constrained(cls, X, centroids, must_link, cannot_link, verbose=False):
        """
        Assign clusters with respect to must-link and cannot-link constraints.

        Parameters:
        X (numpy.ndarray): Data points.
        centroids (numpy.ndarray): Current centroids.
        must_link (dict): Dictionary of must-link constraints.
        cannot_link (dict): Dictionary of cannot-link constraints.

        Returns:
        numpy.ndarray: Cluster assignments.
        """
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=int)

        for i in tqdm(range(n_samples), desc="Assign cluster index", disabel=not verbose):
            # Calculate distances for the current point to all centroids
            distances_to_centroids = np.linalg.norm(X[i] - centroids, axis=1)

            # Sort centroids by distance
            closest_centroids = np.argsort(distances_to_centroids)

            for centroid_index in closest_centroids:
                can_assign = True

                # Check must-link constraints
                if i in must_link:
                    linked_points = must_link[i]
                    if not all(assignments[j] == centroid_index or assignments[j] == 0 for j in linked_points):
                        can_assign = False

                # Check cannot-link constraints
                if can_assign and i in cannot_link:
                    linked_points = cannot_link[i]
                    if any(assignments[j] == centroid_index for j in linked_points):
                        can_assign = False

                # Assign to the closest valid centroid
                if can_assign:
                    assignments[i] = centroid_index
                    break

        return assignments


class CopKMean_Behrouz(SemiSupervisedClustering):
    # from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py

    @classmethod
    def fit_transform(cls, X, k, must_link=None, cannot_link=None, initialization='kmpp', max_iter=300, tol=1e-4,
                      verbose=False):
        """

        :param X:
        :param k:
        :param must_link:
        :param cannot_link:
        :param initialization:
        :param max_iter:
        :param tol:
        :return:
        """
        # Convert constraint lists to dictionaries and resolve transitive closure
        must_link = dict() if must_link is None else must_link
        cannot_link = dict() if cannot_link is None else cannot_link

        if verbose:
            print("optimize must_link info ...")
        must_link_info = cls.grouping_points_with_must_link(must_link, X)

        # Calculate tolerance for convergence
        tol = cls.calculate_tolerance(tol, X)

        if verbose:
            print("initializing centroids...")

        centroids = cls.initialize_centroids(X, k, initialization)
        cluster_assignments = np.zeros(X.shape[0], dtype=int)

        for _ in tqdm(range(max_iter), desc="iterations", disable=not verbose):
            premature_cluster_assignments = cls.assign_clusters_constrained(X, centroids, must_link, cannot_link)

            # Update centroids & cluster_assignment
            cluster_assignments, new_centroids = cls.update_centroids(X, premature_cluster_assignments, k,
                                                                      must_link_info)

            # Check for convergence based on centroids shift
            shift = sum(cls.euclidean_distance(new_centroids[i], centroids[i]) for i in range(k))
            if verbose:
                print("updated centroids with total distance as: {:.3}/{:.3}".format(shift, tol))
            if shift <= tol:
                break
            centroids = new_centroids

        return cluster_assignments, centroids

    @classmethod
    def assign_clusters_constrained(cls, X, centroids, must_link, cannot_link, verbose=False):
        centroids_np = np.array(centroids)
        cluster_assignments = np.full(len(X), -1, dtype=int)

        for i, data_point in tqdm(enumerate(X), desc="assign cluster index", total=len(X), disable=not verbose):
            if cluster_assignments[i] != -1:
                continue

            distances = np.linalg.norm(centroids_np - data_point, axis=1)
            for cluster_idx in np.argsort(distances):
                if not cls.check_violations(i, cluster_idx, cluster_assignments, must_link, cannot_link):
                    cluster_assignments[i] = cluster_idx
                    for linked_idx in must_link[i]:
                        cluster_assignments[linked_idx] = cluster_idx
                    break

            if cluster_assignments[i] == -1:
                return None, None

        return cluster_assignments

    @classmethod
    def update_centroids(cls, X, cluster_assignments, k, must_link_info):
        """
        The update_centroids function is responsible for updating the centroids of clusters
        based on current cluster assignments and handling cases
        where the number of identified clusters (k_new) is less than the intended number of clusters (k).
        The function also incorporates must-link information to adjust the clustering.

        :param X:
        :param cluster_assignments:
        :param k:
        :param must_link_info:
        :return:
        """
        n_features = X.shape[1]
        centers = np.zeros((k, n_features))

        unique_ids, inverse_indices = np.unique(cluster_assignments, return_inverse=True)
        k_real = len(unique_ids)
        for cluster_idx in range(k_real):
            centers[cluster_idx, :] = X[cluster_assignments == unique_ids[cluster_idx]].mean(axis=0)

        if k_real < k:
            must_link_groups, must_link_scores, must_link_centroids = must_link_info
            must_link_centroids_np = np.array(must_link_centroids)
            current_scores = np.array([
                np.sum(np.linalg.norm(centers[inverse_indices[i]] - X[i], axis=1)) for group in must_link_groups
                for i in group
            ])
            group_ids = np.argsort(current_scores - must_link_scores)[::-1]

            for j in range(k - k_real):
                gid = group_ids[j]
                cid = k_real + j
                centers[cid] = must_link_centroids_np[gid]
                for i in must_link_groups[gid]:
                    cluster_assignments[i] = cid

        return cluster_assignments, centers

    @classmethod
    def check_violations(cls, data_index, cluster_index, cluster_assignments, must_link, cannot_link):
        for i in must_link[data_index]:
            if cluster_assignments[i] != -1 and cluster_assignments[i] != cluster_index:
                return True

        for i in cannot_link[data_index]:
            if cluster_assignments[i] == cluster_index:
                return True

        return False

    @classmethod
    def grouping_points_with_must_link(cls, must_link, X, verbose=False):
        """
        End result:
            groups = [[0, 1], [3], [4]]
            flags = [False, False, False, False, False]
        :param must_link:
        :param X:
        :return:
        """

        n_samples = X.shape[0]
        ungrouped_flags = np.ones(n_samples, dtype=bool)

        groups = []
        for i in tqdm(range(n_samples), desc="scan all points for grouping", disabel=not verbose):
            if not ungrouped_flags[i]:
                continue
            group = list(must_link[i] | {i})
            groups.append(group)
            ungrouped_flags[group] = False

        group_centroids = [X[group].mean(axis=0) for group in groups]
        group_scores = [np.sum(np.linalg.norm(X[group] - centroid, axis=1)) for group, centroid in
                        zip(groups, group_centroids)]

        return groups, group_scores, group_centroids

    @classmethod
    def labels_to_constraints(cls, label_indices, n_samples):
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
        must_link, cannot_link = super().labels_to_constraints(label_indices)
        for i in range(len(label_indices), n_samples):
            must_link.update({i: set()})
            cannot_link.update({i: set()})
        return must_link, cannot_link
