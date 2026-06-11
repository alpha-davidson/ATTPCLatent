import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def t_SNE_clustering(features, dimension, ax, labels, class_names, plt_colors, perplexity):
    # Initialize configurations
    PERPLEXITY = perplexity
    CLUSTER_DIMENSIONALITY = dimension

    # Standardize completely localized path targets
    master_dir = "./plots/t-sne"
    sub_dir = os.path.join(master_dir, f"{CLUSTER_DIMENSIONALITY}d_plots")
    os.makedirs(sub_dir, exist_ok=True)

    # 1. Apply t-SNE clustering over ALL features at once
    model = TSNE(n_components=CLUSTER_DIMENSIONALITY, perplexity=PERPLEXITY, random_state=42)
    tsne_data = model.fit_transform(features)
    
    # 2. Extract and color external class profiles on the shared axis canvas
    if CLUSTER_DIMENSIONALITY == 2:
        for class_id in range(len(class_names)):
            mask = (labels == class_id)
            if np.any(mask):
                ax.scatter(
                    tsne_data[mask, 0], 
                    tsne_data[mask, 1], 
                    color=plt_colors[class_id], 
                    label=class_names[class_id], 
                    s=15, 
                    alpha=0.7
                )
        plt.savefig(os.path.join(sub_dir, f'plot_perplexity_{PERPLEXITY}.png'), dpi=200)

    elif CLUSTER_DIMENSIONALITY == 3:
        for class_id in range(len(class_names)):
            mask = (labels == class_id)
            if np.any(mask):
                ax.scatter(
                    tsne_data[mask, 0], 
                    tsne_data[mask, 1], 
                    tsne_data[mask, 2], 
                    color=plt_colors[class_id], 
                    label=class_names[class_id], 
                    s=15, 
                    alpha=0.7
                )
        plt.savefig(os.path.join(sub_dir, f'plot_perplexity_{PERPLEXITY}.png'), dpi=200)
        
    return tsne_data 

def UMAP_embedding(features, dimension, ax, labels, class_names, plt_colors, neighbors):
    # Initialize configurations
    NEIGHBORS = neighbors
    CLUSTER_DIMENSIONALITY = dimension

    # Standardize completely localized path targets
    master_dir = "./plots/umap"
    sub_dir = os.path.join(master_dir, f"{CLUSTER_DIMENSIONALITY}d_plots")
    os.makedirs(sub_dir, exist_ok=True)

    # 1. Apply UMAP embedding over ALL features at once
    model = umap.UMAP(n_components=CLUSTER_DIMENSIONALITY, n_neighbors=NEIGHBORS, random_state=42)
    umap_data = model.fit_transform(features)
    
    # 2. Extract and color external class profiles on the shared axis canvas
    if CLUSTER_DIMENSIONALITY == 2:
        for class_id in range(len(class_names)):
            mask = (labels == class_id)
            if np.any(mask):
                ax.scatter(
                    umap_data[mask, 0], 
                    umap_data[mask, 1], 
                    color=plt_colors[class_id], 
                    label=class_names[class_id], 
                    s=15, 
                    alpha=0.7
                )
        plt.savefig(os.path.join(sub_dir, f'plot_neighbors_{NEIGHBORS}.png'), dpi=200)
        
    elif CLUSTER_DIMENSIONALITY == 3:
        for class_id in range(len(class_names)):
            mask = (labels == class_id)
            if np.any(mask):
                ax.scatter(
                    umap_data[mask, 0], 
                    umap_data[mask, 1], 
                    umap_data[mask, 2], 
                    color=plt_colors[class_id], 
                    label=class_names[class_id], 
                    s=15, 
                    alpha=0.7
                )
        plt.savefig(os.path.join(sub_dir, f'plot_neighbors_{NEIGHBORS}.png'), dpi=200)
        
    return umap_data

def k_means_clustering(features, labels, dimension, save_dir, num_samples_to_print=10):
    master_dir = "./plots/k_means" if save_dir is None else save_dir
    sub_dir = os.path.join(master_dir, f"{dimension}d_plots")
    os.makedirs(sub_dir, exist_ok=True)

    # k-means clustering on full feature space
    k = len(np.unique(labels))
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init='auto')
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
        label_names = ["0,1,2 Tracks", "3 Tracks", "4,5 Tracks"] if k == 3 else [f"{i}-track" for i in range(k)]

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
            for i in range(k):
                plt.scatter(reduced_features[labels == i, 0],
                            reduced_features[labels == i, 1],
                            color=colors[i], label=label_names[i], s=5)
            plt.title("True Labels")
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(sub_dir, "kmeans_2d.png"), dpi=200)
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
            for i in range(k):
                ax2.scatter(reduced_features[labels == i, 0],
                            reduced_features[labels == i, 1],
                            reduced_features[labels == i, 2],
                            color=colors[i], label=label_names[i], s=5)
            ax2.set_title("True Labels")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(sub_dir, "kmeans_3d.png"), dpi=200)
            plt.close()

    return features, cluster_labels, indices