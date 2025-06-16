import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def t_SNE_clustering(features, labels, data_file_stem, dimension):
    # initialize properties for t-SNE clustering
    PERPLEXITY = 100
    CLUSTER_DIMENSIONALITY = dimension

    # create a folder for t-SNE clustering
    folder_path = f'./clustering_plots/t_sne'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create a folder for plots of particular dimensionality
    folder_path = f'./clustering_plots/t_sne/{CLUSTER_DIMENSIONALITY}d_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # apply t-SNE clustering
    model = TSNE(n_components = CLUSTER_DIMENSIONALITY, perplexity = PERPLEXITY)
    tsne_data = model.fit_transform(features)
    
    fig = plt.figure()
    
    # plot the clusters in 2D
    if (dimension == 2):
        ax = fig.add_subplot(111)
        
        # extract x- and y-axis values
        x = np.array(tsne_data[:, 0])
        y = np.array(tsne_data[:, 1])
    
        # # visualize the data on a 2D scatter plot
        ax.scatter(x, y)
        plt.savefig('./clustering_plots/t_sne/2d_plots/plot_perplexity_{}'.format(PERPLEXITY))

    # plot the clusters in 3D
    if (dimension == 3):
        ax = fig.add_subplot(111, projection='3d')

        # extract x-, y-, and z-axis values
        x = np.array(tsne_data[:, 0])
        y = np.array(tsne_data[:, 1])
        z = np.array(tsne_data[:, 2])
    
        # visualize the data on a 3D plot        
        ax.scatter(x, y, z)
        plt.savefig('./clustering_plots/t_sne/3d_plots/plot_perplexity_{}'.format(PERPLEXITY))

    plt.close()


def k_means_clustering(features, labels, data_file_stem, dimension):    
    # initialize properties for k-means clustering
    CLUSTER_DIMENSIONALITY = dimension
    k = 4

    # create a folder for k-means clustering
    folder_path = f'./clustering_plots/k_means'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create a folder for plots of particular dimensionality
    folder_path = f'./clustering_plots/k_means/{CLUSTER_DIMENSIONALITY}d_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # reduce the dataset to desired dimensionality
    pca = PCA(n_components = CLUSTER_DIMENSIONALITY)
    features = pca.fit_transform(features)
    
    # apply k-means clustering
    kmeans_cluster = KMeans(init = "k-means++", n_clusters = k)
    kmeans_data = kmeans_cluster.fit_transform(features)
    
    # calculate the centroids
    centroids = kmeans_cluster.cluster_centers_
    label = kmeans_cluster.fit_predict(features)
    unique_labels = np.unique(label)

    # plot the clusters in 2D
    if (dimension == 2):
        fig = plt.figure()
        
        # scatter plot for each cluster
        for i in unique_labels:
            plt.scatter(features[label == i, 0],
                        features[label == i, 1],
                        label=f'Cluster {i}')
            
        # scatter plot for centroids
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='k', zorder=10)
        plt.legend()
        plt.savefig(f'clustering_plots/k_means/2d_plots/plot_k_{k}.png')

    # plot the clusters in 3D
    if (dimension == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # scatter plot for each cluster
        for i in unique_labels:
            ax.scatter(features[label == i, 0],
                       features[label == i, 1],
                       features[label == i, 2],
                       label=f'Cluster {i}')
        
        # scatter plot for centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   marker='x', s=169, linewidths=3,
                   color='k', zorder=10, label='Centroids')
        ax.legend()
        plt.savefig(f'clustering_plots/k_means/3d_plots/plot_k_{k}.png')
    
    plt.close()