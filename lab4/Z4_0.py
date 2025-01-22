import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import DistanceMetric
import matplotlib.pyplot as plt

SEED=1410
metric = DistanceMetric.get_metric('euclidean')

def k_means(X, n_clusters, max_iter=100, tol=0.001):
    np.random.seed(SEED)
    initial_indices = np.random.choice(len(X), n_clusters, replace=False)
    centroids = X[initial_indices]

    for _ in range(max_iter):
        distances = metric.pairwise(X, centroids)
        closest_centroid = np.argmin(distances, axis=1)
        new_centroids = np.array([X[closest_centroid == i].mean(axis=0) for i in range(n_clusters)])
        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return centroids, closest_centroid


def main():
    X, _ = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=SEED)
    n_clusters = 3
    centroids, closest_centroid = k_means(X, n_clusters)


    colors = ['blue', 'green', 'orange']
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X[closest_centroid == i, 0], X[closest_centroid == i, 1], color=colors[i], label=f'Klaster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', label='Centroidy', s=80)
    plt.legend()
    plt.title('Wizualizacja wyników')
    plt.show()

if __name__=='__main__':
    main()
