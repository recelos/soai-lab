from mpi4py import MPI
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_samples = 10000
n_features = 2
n_clusters = 3
max_iter = 100
tol = 0.001

SEED = 1410
np.random.seed(SEED)

if rank == 0:
    X, _ = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, random_state=SEED)
    ave, res = divmod(X.shape[0], size)
    counts = [ave + 1 if p < res else ave for p in range(size)]
    starts = [sum(counts[:p]) for p in range(size)]
    ends = [sum(counts[:p + 1]) for p in range(size)]
    X = [X[starts[p]:ends[p]] for p in range(size)]

    initial_indices = np.random.choice(len(X[0]), n_clusters, replace=False)
    centroids = X[0][initial_indices]
else:
    X = None
    centroids = None

X_local = comm.scatter(X, root=0)
centroids = comm.bcast(centroids, root=0)

for _ in range(max_iter):
    closest_centroid = pairwise_distances_argmin(np.vstack(X_local), centroids)

    new_local_centroids = [X_local[closest_centroid == i].mean(axis=0) if len(X_local[closest_centroid == i]) > 0 else centroids[i] for i in range(n_clusters)]
    gathered_centroids = comm.gather(new_local_centroids, root=0)
    
    if rank == 0:
        aggregated_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            cluster_points = [g[i] for g in gathered_centroids if np.any(g[i])]
            if cluster_points:
                aggregated_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                aggregated_centroids[i] = centroids[i]

        centroids = aggregated_centroids

    centroids = comm.bcast(centroids, root=0)

if rank == 0:
    closest_centroid = pairwise_distances_argmin(np.vstack(X), centroids)
    print(closest_centroid)
    colors = ['blue', 'green', 'orange']
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(np.vstack(X)[closest_centroid == i, 0], np.vstack(X)[closest_centroid == i, 1], color=colors[i], label=f'Klaster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='r', label='Centroidy', s=80)
    plt.legend()
    plt.title('Wizualizacja wyników')
    plt.show()
