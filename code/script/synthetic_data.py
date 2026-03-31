import numpy as np

# generate synthetic data points for each cluster
# dataset is generated synthetically with a fixed random seed for reproducibility
def generate_blobs(centers, n_samples, std, n_features, random_state):
    if random_state is not None:
        np.random.seed(random_state)

    X = []
    y = []

    n_clusters = len(centers)
    samples_per_cluster = n_samples // n_clusters

    for cluster_id, center in enumerate(centers):
        points = np.random.randn(samples_per_cluster, n_features) * std + np.array(center)
        labels = np.full(samples_per_cluster, cluster_id)

        X.append(points)
        y.append(labels)

    X = np.vstack(X) # shape (n_samples, 2)
    y = np.hstack(y) # shape (n_samples, 1)

    return X, y