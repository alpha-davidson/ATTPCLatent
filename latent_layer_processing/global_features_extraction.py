import click
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from pointnet_model import pnet
from clustering import t_SNE_clustering
from clustering import k_means_clustering

@click.command()
@click.option('--beam', default='O16', type=click.STRING, help='The beam to train on (e.g. O16, Mg22, C16)')
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('model-file-stem')
@click.argument('data-file-stem')


def extract_global_features(beam, num_points, num_classes, model_file_stem, data_file_stem):
    # build a model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)
    model.load_weights(model_file_stem)

    # load test data
    BATCH_SIZE = 64
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :4]).batch(BATCH_SIZE)
    test_labels = test_ds[:, :, 4]

    # predict labels
    predicted_probabilities = model.predict(test_features)
    predictions = np.argmax(predicted_probabilities, axis=2)

    # extract global features from latent space
    global_feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
    global_features = global_feature_extractor.predict(test_features)

    # create a folder for results of clustering
    folder_path = "./clustering_plots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    t_SNE_clustering(global_features, test_labels, data_file_stem, 2)
    t_SNE_clustering(global_features, test_labels, data_file_stem, 3)
    k_means_clustering(global_features, test_labels, data_file_stem, 2)
    k_means_clustering(global_features, test_labels, data_file_stem, 3)
    
    return global_features

if __name__ == '__main__':
    extract_global_features()