import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

from utils import (
    validate_features_and_labels,
    validate_plot_dimension,
    get_class_names,
    get_plot_colors,
    plot_labeled_embedding,
)

def t_SNE_clustering(features, dimension, ax, labels, perplexity, class_names=None,
                     plt_colors=None, save_dir=None, plot_name=None, random_state=None):
    """Run t-SNE and plot points colored by class label.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    plt_colors is optional; when omitted, one color is generated per class.
    """

    validate_plot_dimension(dimension, "t_SNE_clustering")
    features, labels = validate_features_and_labels(features, labels)
    labels, unique_labels, class_names = get_class_names(labels, class_names)

    # create a folder for t-SNE clustering
    folder_path = os.path.join(save_dir or "../plots", "t-sne", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f"t_sne_{dimension}d"

    # apply t-SNE clustering over all features at once
    model = TSNE(n_components=dimension, perplexity=perplexity, random_state=random_state)
    tsne_data = model.fit_transform(features)

    plot_labeled_embedding(tsne_data, labels, unique_labels, class_names, ax, plt_colors)
    ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_perplexity_{perplexity}.png"), dpi=200)

    return tsne_data 

def UMAP_embedding(features, dimension, ax, labels, neighbors, class_names=None,
                   plt_colors=None, save_dir=None, plot_name=None, random_state=None):
    """Run UMAP and plot points colored by class label.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    plt_colors is optional; when omitted, one color is generated per class.
    """
    validate_plot_dimension(dimension, "UMAP_embedding")
    features, labels = validate_features_and_labels(features, labels)
    labels, unique_labels, class_names = get_class_names(labels, class_names)

    # create a folder for UMAP embedding
    folder_path = os.path.join(save_dir or "../plots", "umap", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f"umap_{dimension}d"

    # apply UMAP embedding over all features at once
    model = umap.UMAP(n_components=dimension, n_neighbors=neighbors, random_state=random_state)
    umap_data = model.fit_transform(features)

    plot_labeled_embedding(umap_data, labels, unique_labels, class_names, ax, plt_colors)
    ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_neighbors_{neighbors}.png"), dpi=200)

    return umap_data

def k_means_clustering(features, labels, dimension, save_dir=None, num_samples_to_print=10,
                       plot_name=None, class_names=None, random_state=None):
    """Run k-means and compare cluster assignments with known labels.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    """
    validate_plot_dimension(dimension, "k_means_clustering")

    features, labels = validate_features_and_labels(features, labels)
    folder_path = os.path.join(save_dir or "../plots/k_means", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f'k_means_{dimension}d'

    # k-means clustering on full feature space
    labels, unique_labels, class_names = get_class_names(labels, class_names)
    k = len(unique_labels)
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init='auto', random_state=random_state)
    cluster_labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    print("\nRandom sample indices from each cluster:")
    indices = []
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        random_indices = np.random.choice(cluster_indices, 
                                          size=min(num_samples_to_print, len(cluster_indices)), 
                                          replace=False)
        indices.append(random_indices)
        print(f"Cluster {cluster_id}: {random_indices.tolist()}")

    pca = PCA(n_components=dimension)
    reduced_features = pca.fit_transform(features)
    cluster_names = [f"Cluster {i}" for i in range(k)]
    cluster_values = np.arange(k)
    centroid_features = pca.transform(centroids)

    if dimension == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plot_labeled_embedding(reduced_features, cluster_labels, cluster_values, cluster_names, ax1, size=5, alpha=1.0)
        ax1.scatter(centroid_features[:, 0], centroid_features[:, 1],
                    marker='x', s=60, c='black', label='Centroids')
        ax1.set_title("KMeans Clustering")
        ax1.legend()

        plot_labeled_embedding(reduced_features, labels, unique_labels, class_names, ax2, size=5, alpha=1.0)
        ax2.set_title("True Labels")
        ax2.legend()

    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        plot_labeled_embedding(reduced_features, cluster_labels, cluster_values, cluster_names, ax1, size=5, alpha=1.0)
        ax1.scatter(*centroid_features.T, marker='x', s=60, c='black', label='Centroids')
        ax1.set_title("KMeans Clustering")
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        plot_labeled_embedding(reduced_features, labels, unique_labels, class_names, ax2, size=5, alpha=1.0)
        ax2.set_title("True Labels")
        ax2.legend()

    plt.tight_layout()
    filename = f"kmeans_{dimension}d.png" if plot_name is None else f"{plot_name}_kmeans_{dimension}d.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=200)
    plt.close()

    return features, cluster_labels, indices

def pca_clustering(features, labels, dimension, save_dir=None, class_names=None,
                   plot_name=None, random_state=None):
    """Project features with PCA and plot points colored by class label.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    """
    validate_plot_dimension(dimension, "pca_clustering")

    features, labels = validate_features_and_labels(features, labels)
    folder_path = os.path.join(save_dir or "../plots/pca", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    n_components = min(dimension, features.shape[0], features.shape[1])
    if n_components != dimension:
        warnings.warn(
            f"Requested {dimension} PCA components, but only {n_components} are possible "
            f"for features with shape {features.shape}.",
            UserWarning,
        )

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(features)

    labels, unique_labels, class_names = get_class_names(labels, class_names)

    if dimension == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_labeled_embedding(reduced_features, labels, unique_labels, class_names, ax, size=5, alpha=1.0)
        ax.set_title("PCA Projection")
        ax.legend()
        filename = "pca_2d.png" if plot_name is None else f"{plot_name}_pca_2d.png"
        plt.savefig(os.path.join(folder_path, filename), dpi=200)
        plt.close()

    elif dimension == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_labeled_embedding(reduced_features, labels, unique_labels, class_names, ax, size=5, alpha=1.0)
        ax.set_title("PCA Projection")
        ax.legend()
        filename = "pca_3d.png" if plot_name is None else f"{plot_name}_pca_3d.png"
        plt.savefig(os.path.join(folder_path, filename), dpi=200)
        plt.close()

    return reduced_features


def pca_variance_analysis(features, variance_threshold=0.95, save_dir=None,
                          plot_name=None, random_state=None):
    folder_path = os.path.join(save_dir or "../plots/pca", "variance")
    os.makedirs(folder_path, exist_ok=True)

    n_components = min(features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(features)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_needed = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    variance_at_threshold = float(cumulative_variance[n_needed - 1])

    print(
        f"\nPCA variance analysis: {n_needed} component(s) explain "
        f"{variance_at_threshold:.1%} of variance (threshold={variance_threshold:.0%})"
    )

    plt.figure(figsize=(7, 5))
    component_numbers = np.arange(1, n_components + 1)
    plt.plot(component_numbers, cumulative_variance, marker="o", markersize=3)
    plt.axhline(variance_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(n_needed, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = "cumulative_variance.png" if plot_name is None else f"{plot_name}_cumulative_variance.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=200)
    plt.close()

    return {
        "n_components": n_needed,
        "variance_threshold": variance_threshold,
        "variance_explained": variance_at_threshold,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
    }
