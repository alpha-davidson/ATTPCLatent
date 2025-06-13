import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pointnet_model import pnet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def t_SNE_clustering(features, labels, data_file_stem):
    # initialize properties for t-SNE clustering
    PERPLEXITY = 40
    CLUSTER_DIMENSIONALITY = 2
    
    # apply t-SNE clustering
    model = TSNE(n_components = CLUSTER_DIMENSIONALITY, perplexity = PERPLEXITY)
    tsne_data = model.fit_transform(features)

    # extract x- and y-axis values
    x = np.array(tsne_data[:, 0])
    y = np.array(tsne_data[:, 1])

    # visualize the data on a 2D scatter plot
    plt.figure()
    plt.scatter(x, y)
    plt.savefig('tsne_plot_perplexity_{}'.format(PERPLEXITY))

    print(tsne_data.shape)

def k_means_clustering(features, labels, data_file_stem):
    # initialize properties for k-means clustering
    CLUSTER_DIMENSIONALITY = 3
    k = 3

    # reducing the dataset to desired dimensionality
    pca = PCA(n_components = CLUSTER_DIMENSIONALITY)
    features = pca.fit_transform(features)
    
    # apply k-means clustering
    kmeans_cluster = KMeans(init = "k-means++", n_clusters = k)
    kmeans_data = kmeans_cluster.fit_transform(features)
    
    # calculating the centroids
    centroids = kmeans_cluster.cluster_centers_
    label = kmeans_cluster.fit_predict(features)
    unique_labels = np.unique(label)

    
    # plotting the clusters in 2D
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
    plt.savefig(f'kmeans_plot_k_{k}_2d.png')

    
   # plotting the clusters in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # scatter plot for each cluster
    # for i in unique_labels:
    #     ax.scatter(features[label == i, 0],
    #                features[label == i, 1],
    #                features[label == i, 2],
    #                label=f'Cluster {i}')
    
    # # scatter plot for centroids
    # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #            marker='x', s=169, linewidths=3,
    #            color='k', zorder=10, label='Centroids')
    # ax.legend()
    # plt.savefig(f'kmeans_plot_k_{k}_3d.png')

    
    plt.close()

if __name__ == '__main__':
    t_SNE_clustering()