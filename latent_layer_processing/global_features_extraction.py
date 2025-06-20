import os
import sys
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..pointnet import create_pointnet_model
from clustering import t_SNE_clustering
from clustering import k_means_clustering
import matplotlib.pyplot as plt

def labels_extraction(data_file_stem, BATCH_SIZE):
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :4]).batch(BATCH_SIZE)
    test_labels = test_ds[:, :, 4]
    return test_features, test_labels

def global_features(model, data_file_stem, BATCH_SIZE):
    test_features, test_labels = labels_extraction(data_file_stem, BATCH_SIZE)
    global_features = model.predict(test_features)
    return global_features, test_labels

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
    
    # create a folder for results of clustering
    folder_path = "./clustering_plots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # load test data
    global_features_original, test_labels_original = global_features(global_feature_extractor, data_file_stem, BATCH_SIZE)
    global_features_2Body, test_labels_2Body = global_features(global_feature_extractor, '../simulated_data/voxel_data/output_digi_HDF_2Body/O16_size512', BATCH_SIZE)
    global_features_2Body_2T, test_labels_2Body_2T = global_features(global_feature_extractor, '../simulated_data/voxel_data/output_digi_HDF_2Body_2T/O16_size512', BATCH_SIZE)

    # plot t-SNE clustering of experimental, 2Body, and 2Body_2T global features in 2D space
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t_SNE_clustering(global_features_2Body, test_labels_2Body, '../simulated_data/voxel_data/output_digi_HDF_2Body/O16_size512', 2, ax, 'red', 'output_digi_HDF_2Body', 1)
    t_SNE_clustering(global_features_2Body_2T, test_labels_2Body_2T, '../simulated_data/voxel_data/output_digi_HDF_2Body_2T/O16_size512', 2, ax, 'green', 'output_digi_HDF_2Body_2T', 1)
    t_SNE_clustering(global_features_original, test_labels_original, data_file_stem, 2, ax, 'blue', 'experimental', 0.3)
    ax.legend()
    plt.savefig('./clustering_plots/t_sne/2d_plots/plot2')
    plt.close()   

    # plot t-SNE clustering of experimental, 2Body, and 2Body_2T global features in 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t_SNE_clustering(global_features_2Body, test_labels_2Body, '../simulated_data/voxel_data/output_digi_HDF_2Body/O16_size512', 3, ax, 'red', 'output_digi_HDF_2Body', 1)
    t_SNE_clustering(global_features_2Body_2T, test_labels_2Body_2T, '../simulated_data/voxel_data/output_digi_HDF_2Body_2T/O16_size512', 3, ax, 'green', 'output_digi_HDF_2Body_2T', 1)
    t_SNE_clustering(global_features_original, test_labels_original, data_file_stem, 3, ax, 'blue', 'experimental', 0.1)
    ax.legend()
    plt.savefig('./clustering_plots/t_sne/3d_plots/plot2')
    plt.close()   

if __name__ == '__main__':
    extract_global_features()