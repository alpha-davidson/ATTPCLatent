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
    
    # load global features
    global_features_2track, test_labels_2track = global_features(global_feature_extractor, '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512', BATCH_SIZE)
    global_features_3track, test_labels_3track = global_features(global_feature_extractor, '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_3Body/O16_size512', BATCH_SIZE)
    global_features_Mg22, test_labels__Mg22 = global_features(global_feature_extractor, '../simulated_data/process_data/Mg22/voxel_data/output_digi_HDF_Mg22_Ne20pp_8MeV/Mg22_size512', BATCH_SIZE)

    # create a folder to save global features
    folder_path = "./global_features"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # save global features
    np.save(f'./global_features/Mg22_features.npy', global_features_Mg22)
    np.save(f'./global_features/O16_2track_features.npy', global_features_2track)
    np.save(f'./global_features/O16_3track_features.npy', global_features_3track)

if __name__ == '__main__':
    extract_global_features()