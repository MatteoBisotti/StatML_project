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

    def initialize_centroids(self, X):
        """
        Randomly choose K distincs points from X as initial centroids
        """
        n_samples = len(X)

        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()
        return centroids
    
    def euclidean_distances(self, X, centroids):
        """
        Compute squared Euclidean distances from each point to each centroid

        Returns: distances[i, j] = ||X[i] - centroids[j]||^2
        """
        distances = []

        for x in X:
            row = []
            for c in centroids:
                row.append(np.sum((x-c) ** 2))
            distances.append(row)
        
        return np.array(distances)
    
    def assign_clusters(self, X, centroids):
        """
        Assign each point to the nearest centroid
        """
        distances = self.euclidean_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X, labels, old_centroids):
        """
        Update centroids as the mean of assigned points
        """
        n_features = X.shape[1]
        new_centroids = np.empty((self.n_clusters, n_features), dtype=float)

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points)==0:
                new_centroids[k] = old_centroids[k]
            else:
                new_centroids[k] = np.mean(cluster_points, axis=0)
        
        return new_centroids
    
    def compute_inertia(self, X, labels, centroids):
        """
        Compute the inertia 
        """
