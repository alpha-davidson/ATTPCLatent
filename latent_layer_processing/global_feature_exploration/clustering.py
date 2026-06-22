import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import matplotlib.cm as cm

def _get_class_names(labels, class_names):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [str(label) for label in unique_labels]
    if len(class_names) != len(unique_labels):
        raise ValueError(
            f"class_names must have {len(unique_labels)} entries, got {len(class_names)}"
        )
    return labels, unique_labels, class_names

def t_SNE_clustering(features, dimension, ax, labels, perplexity, class_names=None,
                     plt_colors=None, save_dir=None,
                     plot_name=None, random_state=None):
    # initialize properties for t-SNE clustering
    PERPLEXITY = perplexity
    CLUSTER_DIMENSIONALITY = dimension
    labels, unique_labels, class_names = _get_class_names(labels, class_names)
    if plt_colors is None:
        plt_colors = ["red", "blue", "green"]

    # create a folder for t-SNE clustering
    folder_path = os.path.join(save_dir or "../plots", "t-sne", f"{CLUSTER_DIMENSIONALITY}d_plots")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if plot_name is None:
        plot_name = f"t_sne_{CLUSTER_DIMENSIONALITY}d"

    # apply t-SNE clustering over all features at once
    model = TSNE(n_components=CLUSTER_DIMENSIONALITY, perplexity=PERPLEXITY, random_state=random_state)
    tsne_data = model.fit_transform(features)

    # plot the clusters in 2D
    if (dimension == 2):
        for class_index, label_value in enumerate(unique_labels):
            mask = (labels == label_value)
            if np.any(mask):
                ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                           color=plt_colors[class_index % len(plt_colors)], label=class_names[class_index],
                           s=15, alpha=0.7)
        ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_perplexity_{PERPLEXITY}.png"), dpi=200)

    # plot the clusters in 3D
    if (dimension == 3):
        for class_index, label_value in enumerate(unique_labels):
            mask = (labels == label_value)
            if np.any(mask):
                ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1], tsne_data[mask, 2],
                           color=plt_colors[class_index % len(plt_colors)], label=class_names[class_index],
                           s=15, alpha=0.7)
        ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_perplexity_{PERPLEXITY}.png"), dpi=200)

    return tsne_data 

def UMAP_embedding(features, dimension, ax, labels, neighbors, class_names=None,
                   plt_colors=None, save_dir=None,
                   plot_name=None, random_state=None):
    # initialize properties for UMAP embedding
    NEIGHBORS = neighbors
    CLUSTER_DIMENSIONALITY = dimension
    labels, unique_labels, class_names = _get_class_names(labels, class_names)
    if plt_colors is None:
        plt_colors = ["red", "blue", "green"]

    # create a folder for UMAP embedding
    folder_path = os.path.join(save_dir or "../plots", "umap", f"{CLUSTER_DIMENSIONALITY}d_plots")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if plot_name is None:
        plot_name = f"umap_{CLUSTER_DIMENSIONALITY}d"

    # apply UMAP embedding over all features at once
    model = umap.UMAP(n_components=CLUSTER_DIMENSIONALITY, n_neighbors=NEIGHBORS, random_state=random_state)
    umap_data = model.fit_transform(features)

    # plot the clusters in 2D
    if (dimension == 2):
        for class_index, label_value in enumerate(unique_labels):
            mask = (labels == label_value)
            if np.any(mask):
                ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                           color=plt_colors[class_index % len(plt_colors)], label=class_names[class_index],
                           s=15, alpha=0.7)
        ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_neighbors_{NEIGHBORS}.png"), dpi=200)

    # plot the clusters in 3D
    if (dimension == 3):
        for class_index, label_value in enumerate(unique_labels):
            mask = (labels == label_value)
            if np.any(mask):
                ax.scatter(umap_data[mask, 0], umap_data[mask, 1], umap_data[mask, 2],
                           color=plt_colors[class_index % len(plt_colors)], label=class_names[class_index],
                           s=15, alpha=0.7)
        ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_neighbors_{NEIGHBORS}.png"), dpi=200)

    return umap_data

def k_means_clustering(features, labels, dimension, save_dir=None, num_samples_to_print=10,
                       plot_name=None, class_names=None, random_state=None):
    folder_path = os.path.join(save_dir or "../plots/k_means", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f'k_means_{dimension}d'

    # k-means clustering on full feature space
    labels, unique_labels, class_names = _get_class_names(labels, class_names)
    k = len(unique_labels)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=random_state, n_init='auto')
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

    if dimension in [2, 3]:
        pca = PCA(n_components=dimension)
        reduced_features = pca.fit_transform(features)

        color_map = cm.get_cmap('tab10', k) if hasattr(cm, 'get_cmap') else plt.colormaps['tab10'].resampled(k)
        colors = [color_map(i) for i in range(k)]

        if dimension == 2:
            plt.figure(figsize=(14, 6))

            # clustered
            plt.subplot(1, 2, 1)
            for i in range(k):
                plt.scatter(reduced_features[cluster_labels == i, 0],
                            reduced_features[cluster_labels == i, 1],
                            color=colors[i], label=f'Cluster {i}', s=5)
            plt.scatter(pca.transform(centroids)[:, 0],
                        pca.transform(centroids)[:, 1],
                        marker='x', s=60, c='black', label='Centroids')
            plt.title("KMeans Clustering")
            plt.legend()

            # true labels
            plt.subplot(1, 2, 2)
            for i, label_value in enumerate(unique_labels):
                plt.scatter(reduced_features[labels == label_value, 0],
                            reduced_features[labels == label_value, 1],
                            color=colors[i], label=class_names[i], s=5)
            plt.title("True Labels")
            plt.legend()

            plt.tight_layout()
            filename = "kmeans_2d.png" if plot_name is None else f"{plot_name}_kmeans_2d.png"
            plt.savefig(os.path.join(folder_path, filename), dpi=200)
            plt.close()

        elif dimension == 3:
            fig = plt.figure(figsize=(14, 6))

            # clustered
            ax1 = fig.add_subplot(121, projection='3d')
            for i in range(k):
                ax1.scatter(reduced_features[cluster_labels == i, 0],
                            reduced_features[cluster_labels == i, 1],
                            reduced_features[cluster_labels == i, 2],
                            color=colors[i], label=f'Cluster {i}', s=5)
            ax1.scatter(*pca.transform(centroids).T, marker='x', s=60, c='black', label='Centroids')
            ax1.set_title("KMeans Clustering")
            ax1.legend()

            # true labels
            ax2 = fig.add_subplot(122, projection='3d')
            for i, label_value in enumerate(unique_labels):
                ax2.scatter(reduced_features[labels == label_value, 0],
                            reduced_features[labels == label_value, 1],
                            reduced_features[labels == label_value, 2],
                            color=colors[i], label=class_names[i], s=5)
            ax2.set_title("True Labels")
            ax2.legend()

            plt.tight_layout()
            filename = "kmeans_3d.png" if plot_name is None else f"{plot_name}_kmeans_3d.png"
            plt.savefig(os.path.join(folder_path, filename), dpi=200)
            plt.close()

    return features, cluster_labels, indices

def pca_clustering(features, labels, dimension, save_dir=None, class_names=None,
                   plot_name=None, random_state=None):
    folder_path = os.path.join(save_dir or "../plots/pca", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    n_components = min(dimension, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(features)

    labels, unique_labels, class_names = _get_class_names(labels, class_names)
    k = len(unique_labels)
    color_map = cm.get_cmap('tab10', k) if hasattr(cm, 'get_cmap') else plt.colormaps['tab10'].resampled(k)
    colors = [color_map(i) for i in range(k)]

    if dimension == 2:
        plt.figure(figsize=(7, 6))
        for i, label_value in enumerate(unique_labels):
            plt.scatter(reduced_features[labels == label_value, 0],
                        reduced_features[labels == label_value, 1],
                        color=colors[i], label=class_names[i], s=5)
        plt.title("PCA Projection")
        plt.legend()
        filename = "pca_2d.png" if plot_name is None else f"{plot_name}_pca_2d.png"
        plt.savefig(os.path.join(folder_path, filename), dpi=200)
        plt.close()

    elif dimension == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, label_value in enumerate(unique_labels):
            ax.scatter(reduced_features[labels == label_value, 0],
                       reduced_features[labels == label_value, 1],
                       reduced_features[labels == label_value, 2],
                       color=colors[i], label=class_names[i], s=5)
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
