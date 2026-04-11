import matplotlib.pyplot as plt
import numpy as np

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

def plot_results_kmeans_kmeanspp(
        X,
        inertias_kmeans,
        n_iters_kmeans,
        labels_kmeans,
        centroids_kmeans,
        inertias_kmeanspp,
        n_iters_kmeanspp,
        labels_kmeanspp,
        centroids_kmeanspp
):
    print("\nK-Means")
    print("--------")
    print(f"Mean inertia    : {np.mean(inertias_kmeans):.4f}")
    print(f"Std inertia     : {np.std(inertias_kmeans):.4f}")
    print(f"Min inertia     : {np.min(inertias_kmeans):.4f}")
    print(f"Max inertia     : {np.max(inertias_kmeans):.4f}")
    print(f"Mean iterations : {np.mean(n_iters_kmeans):.2f}")
    print(f"Max iterations K-Means    : {np.max(n_iters_kmeans)}")

    print("\nK-Means++")
    print("-----------")
    print(f"Mean inertia    : {np.mean(inertias_kmeanspp):.4f}")
    print(f"Std inertia     : {np.std(inertias_kmeanspp):.4f}")
    print(f"Min inertia     : {np.min(inertias_kmeanspp):.4f}")
    print(f"Max inertia     : {np.max(inertias_kmeanspp):.4f}")
    print(f"Mean iterations : {np.mean(n_iters_kmeanspp):.2f}")
    print(f"Max iterations K-Means    : {np.max(n_iters_kmeans)}")

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(inertias_kmeans, bins=15, alpha=0.6, label="K-Means")
    plt.hist(inertias_kmeanspp, bins=15, alpha=0.6, label="K-Means++")

    plt.xlabel("Inertia")
    plt.ylabel("Frequency")
    plt.title("Distribution of inertia")
    plt.legend()
    plt.tight_layout()
    plt.show()

    '''best_kmeans_idx = np.argmin(inertias_kmeans)
    best_kmeanspp_idx = np.argmin(inertias_kmeanspp)

    # K-Means best
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans[best_kmeans_idx], s=25)
    plt.scatter(
        centroids_kmeans[best_kmeans_idx][:, 0],
        centroids_kmeans[best_kmeans_idx][:, 1],
        marker="X",
        s=200
    )
    plt.title("Best run - K-Means")
    plt.tight_layout()
    plt.show()

    # K-Means++ best
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_kmeanspp[best_kmeanspp_idx], s=25)
    plt.scatter(
        centroids_kmeanspp[best_kmeanspp_idx][:, 0],
        centroids_kmeanspp[best_kmeanspp_idx][:, 1],
        marker="X",
        s=200
    )
    plt.title("Best run - K-Means++")
    plt.tight_layout()
    plt.show()'''

    # boxplot 
    plt.figure(figsize=(6, 5))
    plt.boxplot([inertias_kmeans, inertias_kmeanspp],
                labels=["K-Means", "K-Means++"])

    plt.ylabel("Inertia")
    plt.title("Inertia comparison")
    plt.tight_layout()
    plt.show()