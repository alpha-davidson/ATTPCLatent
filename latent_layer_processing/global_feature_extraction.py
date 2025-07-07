import os
import sys
import click
import numpy as np
import random
import pandas as pd
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

def extraction(data_file_stem, BATCH_SIZE):
    ds = np.load('{}{}'.format(data_file_stem, '.npy'))
    features = tf.data.Dataset.from_tensor_slices(ds[:, :, :4]).batch(BATCH_SIZE)
    return features

def global_features(model, data_file_stem, BATCH_SIZE):
    features = extraction(data_file_stem, BATCH_SIZE)
    global_features = model.predict(features)
    return global_features

def test_train_split(ISOTOPE, global_features):
    # generates an array of numbers as long as the length of the data to randomize the events 
    name = ISOTOPE + "global_features"
    all_events = global_features
    rand_shuffle = np.random.choice(len(all_events), len(all_events), replace = False)
    
    # 20% for test
    test_split = int(len(all_events) * .2)
    
    test_data =  all_events[rand_shuffle[:test_split],:]    #only saving the indices and number of tracks of the test events
    train_data = all_events[rand_shuffle[test_split:],:]

    folder_path = f"./global_features/{ISOTOPE}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    print(test_data.shape, train_data.shape)
    np.save(f'./global_features/{ISOTOPE}/test', test_data)
    np.save(f'./global_features/{ISOTOPE}/train', train_data)
    
    
@click.command()
@click.argument('model-folder')
    
def extract_global_features(model_folder):
    """
    Sample invocation:
PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.global_features_extraction ../training/O16_models/2025-06-16-14:56:34/full_model
    """
    # load model
    model = tf.keras.models.load_model(model_folder)    
    BATCH_SIZE = 32

    # extract global features from latent space
    global_feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)

    # extract global features
    data_file_O16_2track = '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512'
    data_file_O16_2track_2t = '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body_2T/O16_size512'
    data_file_O16_3track = '../simulated_data/process_data/O16/voxel_data/output_digi_HDF_3Body/O16_size512'
    data_file_O16_Exp = '../data_processing/O16/voxel_data/O16_size512'
    
    global_features_2track = global_features(global_feature_extractor, data_file_O16_2track, BATCH_SIZE)
    global_features_2track_2t = global_features(global_feature_extractor, data_file_O16_2track_2t, BATCH_SIZE)
    global_features_3track = global_features(global_feature_extractor, data_file_O16_3track, BATCH_SIZE)
    experimental_features_O16 = global_features(global_feature_extractor, data_file_O16_Exp, BATCH_SIZE)

    print(global_features_2track.shape)
    # create a folder to save global features
    folder_path = "./global_features"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # save global features
    dir_O16_2track = './global_features/O16_2Track.npy'
    dir_file_O16_2track_2t = './global_features/O16_2Track_2T.npy'
    dir_file_O16_3track = './global_features/O16_3Track.npy'
    dir_file_O16_Exp = './global_features/O16_experimental_features.npy'
    
    np.save(dir_O16_2track, global_features_2track)
    np.save(dir_file_O16_2track_2t, global_features_2track_2t)
    np.save(dir_file_O16_3track, global_features_3track)
    np.save(dir_file_O16_Exp, experimental_features_O16)

    # *** Comment if needed *** extract labels for O16 Exp in the format event : label   
    path_to_O16_labels = 'O16_labels.csv'
    df = pd.read_csv(path_to_O16_labels, usecols=[0, 1])
    df = df.dropna(subset=['Number of tracks'])  # adjust column name if needed
    df['Number of tracks'] = df['Number of tracks'].astype(int)
    labels = df.to_numpy()
    dir_to_extracted_labels = 'O16_Experimental_Labels.npy'
    np.save(dir_to_extracted_labels, labels)

if __name__ == '__main__':
    extract_global_features()