import matplotlib.pyplot as plt
import numpy as np


# plot synthetic points 
def plot_synthetic_points(X, y, title):
    plt.figure(figsize=(12,8))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# print kmeans and kmeans++ results
def results_kmeans_kmeanspp(
        X,
        inertias_kmeans,
        n_iters_kmeans,
        inertias_kmeanspp,
        n_iters_kmeanspp,
):
    print("\nK-Means")
    print("--------")
    print(f"Mean inertia    : {np.mean(inertias_kmeans):.4f}")
    print(f"Std inertia     : {np.std(inertias_kmeans):.4f}")
    print(f"Min inertia     : {np.min(inertias_kmeans):.4f}")
    print(f"Max inertia     : {np.max(inertias_kmeans):.4f}")
    print(f"Mean iterations : {np.mean(n_iters_kmeans):.2f}")

    print("\nK-Means++")
    print("-----------")
    print(f"Mean inertia    : {np.mean(inertias_kmeanspp):.4f}")
    print(f"Std inertia     : {np.std(inertias_kmeanspp):.4f}")
    print(f"Min inertia     : {np.min(inertias_kmeanspp):.4f}")
    print(f"Max inertia     : {np.max(inertias_kmeanspp):.4f}")
    print(f"Mean iterations : {np.mean(n_iters_kmeanspp):.2f}")


# plot boxplot on number of iterations
def plot_iterations(n_iter_kmeans, n_iter_kmeanspp):
    plt.figure(figsize=(6, 5))
    plt.boxplot([n_iter_kmeans, n_iter_kmeanspp],
                tick_labels=["K-Means", "K-Means++"])
    plt.ylabel("Number of iterations")
    plt.title("Iterations to convergence")
    plt.ylim(0)
    plt.tight_layout()
    plt.show()


# plot strip plot to visualize and compare the distribution of inertias
def plot_strip(inertias_kmeans, inertias_kmeanspp):
    rng = np.random.default_rng(42)

    plt.figure(figsize=(6, 5))

    x_kmeans = rng.normal(0, 0.03, size=len(inertias_kmeans))
    x_kmeanspp = rng.normal(1, 0.03, size=len(inertias_kmeanspp))

    plt.scatter(x_kmeans, inertias_kmeans, s=30, label="K-Means")
    plt.scatter(x_kmeanspp, inertias_kmeanspp, s=30, label="K-Means++")
    plt.xticks([0, 1], ["K-Means", "K-Means++"])
    plt.ylabel("Inertia")
    plt.title("Inertia values across runs")
    plt.tight_layout()
    plt.show()


# plot best and worst run for kmeans and kmeans++
def plot_best_worst_comparison(
    X,
    inertias_kmeans,
    labels_kmeans,
    centroids_kmeans,
    inertias_kmeanspp,
    labels_kmeanspp,
    centroids_kmeanspp
):
    best_kmeans_idx = np.argmin(inertias_kmeans)
    worst_kmeans_idx = np.argmax(inertias_kmeans)

    best_kmeanspp_idx = np.argmin(inertias_kmeanspp)
    worst_kmeanspp_idx = np.argmax(inertias_kmeanspp)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # K-Means best
    axes[0, 0].scatter(
        X[:, 0], X[:, 1],
        c=labels_kmeans[best_kmeans_idx],
        s=25,
        alpha=0.8
    )
    axes[0, 0].scatter(
        centroids_kmeans[best_kmeans_idx][:, 0],
        centroids_kmeans[best_kmeans_idx][:, 1],
        marker="X",
        s=220,
        linewidths=2
    )
    axes[0, 0].set_title(
        f"K-Means - Best run (Inertia = {inertias_kmeans[best_kmeans_idx]:.2f})"
    )
    axes[0, 0].set_xlabel("Feature 1")
    axes[0, 0].set_ylabel("Feature 2")

    # K-Means worst
    axes[0, 1].scatter(
        X[:, 0], X[:, 1],
        c=labels_kmeans[worst_kmeans_idx],
        s=25,
        alpha=0.8
    )
    axes[0, 1].scatter(
        centroids_kmeans[worst_kmeans_idx][:, 0],
        centroids_kmeans[worst_kmeans_idx][:, 1],
        marker="X",
        s=220,
        linewidths=2
    )
    axes[0, 1].set_title(
        f"K-Means - Worst run (Inertia = {inertias_kmeans[worst_kmeans_idx]:.2f})"
    )
    axes[0, 1].set_xlabel("Feature 1")
    axes[0, 1].set_ylabel("Feature 2")

    # K-Means++ best
    axes[1, 0].scatter(
        X[:, 0], X[:, 1],
        c=labels_kmeanspp[best_kmeanspp_idx],
        s=25,
        alpha=0.8
    )
    axes[1, 0].scatter(
        centroids_kmeanspp[best_kmeanspp_idx][:, 0],
        centroids_kmeanspp[best_kmeanspp_idx][:, 1],
        marker="X",
        s=220,
        linewidths=2
    )
    axes[1, 0].set_title(
        f"K-Means++ - Best run (Inertia = {inertias_kmeanspp[best_kmeanspp_idx]:.2f})"
    )
    axes[1, 0].set_xlabel("Feature 1")
    axes[1, 0].set_ylabel("Feature 2")

    # K-Means++ worst
    axes[1, 1].scatter(
        X[:, 0], X[:, 1],
        c=labels_kmeanspp[worst_kmeanspp_idx],
        s=25,
        alpha=0.8
    )
    axes[1, 1].scatter(
        centroids_kmeanspp[worst_kmeanspp_idx][:, 0],
        centroids_kmeanspp[worst_kmeanspp_idx][:, 1],
        marker="X",
        s=220,
        linewidths=2
    )
    axes[1, 1].set_title(
        f"K-Means++ - Worst run (Inertia = {inertias_kmeanspp[worst_kmeanspp_idx]:.2f})"
    )
    axes[1, 1].set_xlabel("Feature 1")
    axes[1, 1].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


# plot boxplot on inertias 
def boxplot(inertias_kmeans, inertias_kmeanspp):
    plt.figure(figsize=(6,5))

    plt.boxplot([inertias_kmeans, inertias_kmeanspp], labels=['K-Means', 'K-Means++'])

    plt.show()