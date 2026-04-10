import numpy as np

class Kmeans:
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
            raise ValueError("tol must be a positive number.")

        self.centroids = None
        self.n_iter = None
        self.labels = None
        self.inertia = None

    def initialize_centroids(self, X):
        """
        Randomly choose K distincs points from X as initial centroids
        """
        n_samples = len(X)

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
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
        inertia = 0.0

        for k in range(self.n_clusters):
            clusters_points = X[labels==k]

            if len(clusters_points) > 0:
                inertia += np.sum((clusters_points - centroids[k])**2)

        return inertia
    
    def fit(self, X):
        """
        Fit K-Means to the data
        """
        X = np.asarray(X, dtype=float)

        centroids = self.initialize_centroids(X)
        flag = False            # convergence flag

        for iter in range(self.max_iter):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels, centroids)

            centroids_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if centroids_shift <= self.tol:
                self.n_iter = iter + 1
                flag = True
                break 
        
        if not flag:
            self.n_iter = self.max_iter

        self.centroids = centroids
        self.labels = self.assign_clusters(X, self.centroids)
        self.inertia = self.compute_inertia(X, self.labels, self.centroids)

        return self
    
    def predict(self, X):
        """
        Assign cluster labels to new data using learned centroids
        """
        if self.centroids is None:
            raise ValueError("The model has not been fitted yet.")
        
        X = np.asarray(X, dtype=float)

        return self.assign_clusters(X, self.centroids)


class KmeansPlusPlus(Kmeans):
    def initialize_centroids(self, X):
        """
        Initialize centroids using the k-means++ strategy
        """
        n_samples = len(X)

        rng = np.random.default_rng(self.random_state)

        first_idx = rng.choice(n_samples)
        centroids = [X[first_idx].copy()]   # first random centroid

        for _ in range(1, self.n_clusters):
            distances = self.euclidean_distances(X, np.array(centroids))

            min_distances = np.min(distances, axis=1)

            total = np.sum(min_distances)

            probabilities = min_distances/total 

            next_idx = rng.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx].copy())
        
        return np.array(centroids)



        

