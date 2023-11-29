import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


class SemiSupervisedGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-3, reg_covar=1e-6):
        """
        Initialize the Semi-Supervised Gaussian Mixture Model with diagonal covariance.

        :param n_components: Number of mixture components.
        :param max_iter: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        :param reg_covar: Regularization added to the diagonal of covariance matrices.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol,
                                   reg_covar=reg_covar, covariance_type='diag')

    def fit(self, X_unlabeled, X_labeled, y_labeled):
        """
        Fit the model using both labeled and unlabeled data.

        :param X_unlabeled: Unlabeled data points.
        :param X_labeled: Labeled data points.
        :param y_labeled: Labels for the labeled data points.
        """
        # Initialize GMM parameters using labeled data
        self._initialize_parameters(X_labeled, y_labeled)

        # Fit the GMM using both labeled and unlabeled data
        X_combined = np.vstack((X_unlabeled, X_labeled))
        self.gmm.fit(X_combined)

    def _initialize_parameters(self, X_labeled, y_labeled):
        """
        Initialize GMM parameters using labeled data with regularization and diagonal covariance.

        :param X_labeled: Labeled data points.
        :param y_labeled: Labels for the labeled data points.
        """
        labels = np.unique(y_labeled)
        if len(labels) != self.n_components:
            raise ValueError("Number of unique labels must equal the number of GMM components.")

        means = np.array([X_labeled[y_labeled == label].mean(axis=0) for label in labels])
        covariances = np.array([np.var(X_labeled[y_labeled == label], axis=0) + self.reg_covar for label in labels])

        self.gmm.means_init = means
        self.gmm.precisions_init = 1.0 / covariances

    def predict(self, X):
        """
        Predict the labels for the given data points.

        :param X: Data points to predict labels for.
        :return: Predicted labels.
        """
        return self.gmm.predict(X)

    def predict_proba(self, X):
        """
        Predict the probabilities for each component for the given data points.

        :param X: Data points to predict probabilities for.
        :return: Predicted probabilities.
        """
        return self.gmm.predict_proba(X)

    def centroids(self):
        return self.gmm.means_


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)

    # Split into labeled and unlabeled data
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.8, random_state=42)

    # Create and fit the semi-supervised GMM
    ssgmm = SemiSupervisedGMM(n_components=3)
    ssgmm.fit(X_unlabeled, X_labeled, y_labeled)

    # Predict labels for unlabeled data
    predicted_labels = ssgmm.predict(X_unlabeled)

    # Check some of the predictions
    print(predicted_labels[:10])
