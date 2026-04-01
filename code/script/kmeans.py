import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter, tol, random_state):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if tol <= 0:
            raise ValueError("tol must be a positive integer.")

        self.centroids = None

        self.rng = np.random.default_rng(random_state)

    def initialize_centroids(self, X):
        """
        Randomly choose K distincs points from X as initial centroids.
        """
        n_samples = len(X)

        indices = self.rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()
        return centroids
