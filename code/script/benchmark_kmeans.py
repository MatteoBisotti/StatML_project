from sklearn.cluster import KMeans
import numpy as np

def benchmark_kmeans(X, n_clusters):
    # multiple runs of KMeans from scikit learn
    inertias_sklearn = []
    n_iters_sklearn = []
    labels_sklearn = []
    centroids_sklearn = []

    inertias_sklearn_pp = []
    n_iters_sklearn_pp = []
    labels_sklearn_pp = []
    centroids_sklearn_pp = []

    # settings
    n_clusters = n_clusters
    n_runs = 100
    max_iter = 100
    tol = 1e-4

    for seed in range(n_runs):
        model_sklearn = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=1,
            max_iter=max_iter,
            tol=tol,
            random_state=seed
        )

        model_sklearn.fit(X)

        inertias_sklearn.append(model_sklearn.inertia_)
        n_iters_sklearn.append(model_sklearn.n_iter_)
        labels_sklearn.append(model_sklearn.labels_.copy())
        centroids_sklearn.append(model_sklearn.cluster_centers_.copy())

        model_sklearn_pp = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=1,
            max_iter=max_iter,
            tol=tol,
            random_state=seed
        )

        model_sklearn_pp.fit(X)

        inertias_sklearn_pp.append(model_sklearn_pp.inertia_)
        n_iters_sklearn_pp.append(model_sklearn_pp.n_iter_)
        labels_sklearn_pp.append(model_sklearn_pp.labels_.copy())
        centroids_sklearn_pp.append(model_sklearn_pp.cluster_centers_.copy())

    inertias_sklearn = np.array(inertias_sklearn)
    n_iters_sklearn = np.array(n_iters_sklearn)

    inertias_sklearn_pp = np.array(inertias_sklearn_pp)
    n_iters_sklearn_pp = np.array(n_iters_sklearn_pp)

    print("\nScikit-learn K-Means")
    print("--------------------")
    print(f"Mean inertia    : {np.mean(inertias_sklearn):.4f}")
    print(f"Std inertia     : {np.std(inertias_sklearn):.4f}")
    print(f"Min inertia     : {np.min(inertias_sklearn):.4f}")
    print(f"Max inertia     : {np.max(inertias_sklearn):.4f}")
    print(f"Mean iterations : {np.mean(n_iters_sklearn):.2f}")

    print("\nScikit-learn K-Means++")
    print("--------------------")
    print(f"Mean inertia    : {np.mean(inertias_sklearn_pp):.4f}")
    print(f"Std inertia     : {np.std(inertias_sklearn_pp):.4f}")
    print(f"Min inertia     : {np.min(inertias_sklearn_pp):.4f}")
    print(f"Max inertia     : {np.max(inertias_sklearn_pp):.4f}")
    print(f"Mean iterations : {np.mean(n_iters_sklearn_pp):.2f}")