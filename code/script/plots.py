import matplotlib.pyplot as plt

# plot synthetic points 
def plot_synthetic_points(X, y):
    plt.figure(figsize=(7, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25)
    plt.title("Synthetic dataset with good clustering")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def plot_kmeans_result(X, labels, centroids, init):
    plt.figure(figsize=(6,6))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=20)
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=100)

    plt.title("KMeans result ("+init+")")
    plt.show()