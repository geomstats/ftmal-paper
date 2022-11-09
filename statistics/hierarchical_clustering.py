"""Plot an Agglomerative Hierarchical Clustering on the sphere."""

import logging
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.agglomerative_hierarchical_clustering import (
    AgglomerativeHierarchicalClustering,
)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = gs.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def main():
    """Plot an Agglomerative Hierarchical Clustering on the sphere."""
    sphere = Hypersphere(dim=2)
    sphere_distance = sphere.metric.dist

    n_clusters = 2
    n_samples_per_dataset = 50

    dataset_1 = sphere.random_von_mises_fisher(
        kappa=10, n_samples=n_samples_per_dataset
    )
    dataset_2 = -sphere.random_von_mises_fisher(
        kappa=10, n_samples=n_samples_per_dataset
    )
    dataset = gs.concatenate((dataset_1, dataset_2), axis=0)

    clustering = AgglomerativeHierarchicalClustering(
        n_clusters=None, distance=sphere_distance, distance_threshold=0,
    )
    clustering.fit(dataset)
    plot_dendrogram(clustering)

    clustering = AgglomerativeHierarchicalClustering(
        n_clusters=n_clusters, distance=sphere_distance
    )
    clustering.fit(dataset)

    clustering_labels = clustering.labels_


    plt.figure(0)
    ax = plt.subplot(111, projection="3d")
    plt.title("Agglomerative Hierarchical Clustering")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    for i_label in range(n_clusters):
        points_label_i = dataset[clustering_labels == i_label, ...]
        sphere_plot.draw_points(ax=ax, points=points_label_i)

    plt.show()


if __name__ == "__main__":
    if os.environ["GEOMSTATS_BACKEND"] == "tensorflow":
        logging.info(
            "Examples with visualizations are only implemented "
            "with numpy backend.\n"
            "To change backend, write: "
            "export GEOMSTATS_BACKEND = 'numpy'."
        )
    else:
        main()
