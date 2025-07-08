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

def t_SNE_clustering(features, dimension, ax, color, label, alpha, perplexity):
    # initialize properties for t-SNE clustering
    PERPLEXITY = perplexity
    CLUSTER_DIMENSIONALITY = dimension

    # create a folder for t-SNE clustering
    folder_path = f'../plots/t_sne'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create a folder for plots of particular dimensionality
    folder_path = f'../plots/t_sne/{CLUSTER_DIMENSIONALITY}d_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # apply t-SNE clustering
    model = TSNE(n_components = CLUSTER_DIMENSIONALITY, perplexity = PERPLEXITY)
    tsne_data = model.fit_transform(features)
    
    # plot the clusters in 2D
    if (dimension == 2):
        # ax = fig.add_subplot(111)
        
        # extract x- and y-axis values
        x = np.array(tsne_data[:, 0])
        y = np.array(tsne_data[:, 1])
    
        # # visualize the data on a 2D scatter plot
        ax.scatter(x, y, color=color, label=label, s=5, alpha=alpha)
        plt.savefig(f'plots/t-sne/2d_plots/plot_perplexity_{PERPLEXITY}.png')

    # plot the clusters in 3D
    if (dimension == 3):
        # extract x-, y-, and z-axis values
        x = np.array(tsne_data[:, 0])
        y = np.array(tsne_data[:, 1])
        z = np.array(tsne_data[:, 2])
    
        # visualize the data on a 3D plot        
        ax.scatter(x, y, z, color=color, label=label, s=5, alpha=alpha)
        plt.savefig(f'plots/t-sne/3d_plots/plot_perplexity_{PERPLEXITY}.png')
    return tsne_data 

def UMAP_embedding(features, dimension, ax, color, label, alpha, neighbors):
    # initialize properties for UMAP embedding
    NEIGHBORS = neighbors
    CLUSTER_DIMENSIONALITY = dimension

    # create a folder for UMAP embedding
    folder_path = f'../plots/umap'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create a folder for plots of particular dimensionality
    folder_path = f'../plots/umap/{CLUSTER_DIMENSIONALITY}d_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # apply UMAP embedding
    model = umap.UMAP(n_components=CLUSTER_DIMENSIONALITY, n_neighbors=NEIGHBORS)
    umap_data = model.fit_transform(features)
    print(umap_data)
    # plot the clusters in 2D
    if (dimension == 2):
        # extract x- and y-axis values
        x = np.array(umap_data[:, 0])
        y = np.array(umap_data[:, 1])
    
        # # visualize the data on a 2D scatter plot
        ax.scatter(x, y, color=color, label=label, s=5, alpha=alpha)
        plt.savefig(f'plots/umap/2d_plots/plot_neighbors_{NEIGHBORS}.png')
    # plot the clusters in 3D
    if (dimension == 3):
        # extract x-, y-, and z-axis values
        x = np.array(umap_data[:, 0])
        y = np.array(umap_data[:, 1])
        z = np.array(umap_data[:, 2])
    
        # visualize the data on a 3D plot        
        ax.scatter(x, y, z, color=color, label=label, s=5, alpha=alpha)
        plt.savefig(f'plots/umap/3d_plots/plot_neighbors_{NEIGHBORS}.png')
    return umap_data

def k_means_clustering(features, labels, dimension, save_dir, num_samples_to_print=10):
    # create a folder for k-means clustering
    folder_path = f'../plots/k_means'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create a folder for plots of particular dimensionality
    folder_path = f'../plots/k_means/{dimension}d_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # k-means clustering on full feature space
    k = len(np.unique(labels))
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    # print random indices for each cluster
    print("\nRandom sample indices from each cluster:")
    indices = []
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        random_indices = np.random.choice(cluster_indices, 
                                          size=min(num_samples_to_print, len(cluster_indices)), 
                                          replace=False)
        indices.append(random_indices)
        print(f"Cluster {cluster_id}: {random_indices.tolist()}")

    # visualization for 2D or 3D
    if dimension in [2, 3]:
        pca = PCA(n_components=dimension)
        reduced_features = pca.fit_transform(features)

        color_map = cm.get_cmap('tab20', k)
        colors = [color_map(i) for i in range(k)]
        label_names = [f"{i}-track" for i in range(k)]

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
            plt.savefig(os.path.join(folder_path, "kmeans_2d.png"))
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
            plt.savefig(os.path.join(folder_path, "kmeans_3d.png"))
            plt.close()

    return features, cluster_labels, indices