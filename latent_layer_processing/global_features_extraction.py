import os
import sys
import click
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from ..pointnet import create_pointnet_model
from ..simulated_data.plot_data import plot_event
from clustering import t_SNE_clustering
from clustering import UMAP_embedding
from clustering import k_means_clustering
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from matplotlib import colors
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler

def labels_extraction(data_file_stem, BATCH_SIZE):
    ds = np.load('{}{}'.format(data_file_stem, '.npy'))
    features = tf.data.Dataset.from_tensor_slices(ds[:, :, :4]).batch(BATCH_SIZE)
    labels = ds[:, :, 4]
    return features, labels

def global_features(model, data_file_stem, BATCH_SIZE):
    features, labels = labels_extraction(data_file_stem, BATCH_SIZE)
    global_features = model.predict(features)
    return global_features, labels

def plot_umap_events(umap_data, name, data_file_stem):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sampled_data = np.load(f'{data_file_stem}.npy')

    folder_path = f'./plots/umap/2d_plots/events_{name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    j = 0
    i = 0
    maxim = 0
    while j != 10:   
        if (maxim == len(umap_data)):
            break
        if (umap_data[i][0] >= 2 and umap_data[i][0] < 7 and umap_data[i][1] >= 2.5 and umap_data[i][1] < 6): # change the boundaries
            fig = plt.figure(figsize=(10, 10))
            plot_event(fig, sampled_data, i)
            plt.savefig(f'./plots/umap/2d_plots/events_{name}/event_{i}.png')
            j+=1
        i = random.randint(0, len(umap_data) - 1)
        maxim+=1
            
@click.command()
@click.option('--beam', default='O16', type=click.STRING, help='The beam to train on (e.g. O16, Mg22, C16)')
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('model-folder')
@click.argument('data-file-stem')
    
def extract_global_features(beam, num_points, num_classes, model_folder, data_file_stem):
    """
    Sample invocation:
PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.global_features_extraction.py --beam O16 --num-classes 24 ../training/O16_models/2025-06-16-14:56:34/full_model ../data_processing/O16/voxel_data/O16_size512
    """
    # load model
    model = tf.keras.models.load_model(model_folder)    
    BATCH_SIZE = 32

    # extract global features from latent space
    global_feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
    
    # load global features (change import files or comment this section if needed)
    global_features_2track, test_labels_2track = global_features(global_feature_extractor, '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512', BATCH_SIZE)
    global_features_3track, test_labels_3track = global_features(global_feature_extractor, '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_3Body/O16_size512', BATCH_SIZE)
    global_features_Mg22, test_labels__Mg22 = global_features(global_feature_extractor, '../simulated_data/process_data/Mg22/voxel_data/output_digi_HDF_Mg22_Ne20pp_8MeV/Mg22_size512', BATCH_SIZE)

    
    # create a folder for results of clustering/embedding
    folder_path = "./plots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    #plot umap embeddings (or comment if needed)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    neighbors = 250
    umap_Mg22_simulated = UMAP_embedding(global_features_Mg22, 2, ax, 'red', 'Mg22 Simulated', 0.5, neighbors)
    umap_2track_O16 = UMAP_embedding(global_features_2track, 2, ax, 'blue', 'O16 2 tracks', 0.5, neighbors)
    umap_3track_O16 = UMAP_embedding(global_features_3track, 2, ax, 'green', 'O16 3 tracks', 0.5, neighbors)
    ax.legend()
    plt.title("UMAP 2D Embedding")

    folder_path = "./plots/umap/2d_plots/umap_data"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(f'./plots/umap/2d_plots/umap_data/Mg22_Simulated_n{neighbors}.npy', umap_Mg22_simulated)
    np.save(f'./plots/umap/2d_plots/umap_data/O16_2track_n{neighbors}.npy', umap_2track_O16)
    np.save(f'./plots/umap/2d_plots/umap_data/O16_3track_n{neighbors}.npy', umap_3track_O16)
    plt.savefig(f'./plots/umap/2d_plots/Mg22+O16_n_{neighbors}.png')
    plt.close()  


    # plot events of defined area of a cluster (change import files or comment if needed)
    umap_Mg22 = np.load('./plots/umap/2d_plots/umap_data/Mg22_Simulated_n200.npy')
    umap_3track = np.load('./plots/umap/2d_plots/umap_data/O16_3track_n200.npy')
    umap_2track = np.load('./plots/umap/2d_plots/umap_data/O16_2track_n200.npy')
    
    plot_umap_events(umap_Mg22, 'Mg22_Simulated', '../simulated_data/process_data/Mg22/voxel_data/output_digi_HDF_Mg22_Ne20pp_8MeV/Mg22_size512')
    plot_umap_events(umap_3track, '3_track', '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_3Body/O16_size512')
    plot_umap_events(umap_2track, '2_track', '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512')

if __name__ == '__main__':
    extract_global_features()