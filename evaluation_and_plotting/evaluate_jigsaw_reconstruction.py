import sys
import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from ..pointnet import create_pointnet_model
from ..plotting import plot_events, plot_histogram, plot_zero_one_bins

@click.command()
@click.option('--beam', default='O16', type=click.STRING, help='The beam to train on (e.g. O16, Mg22, C16)')
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('model-folder')
@click.argument('model-file-stem')
@click.argument('data-file-stem')

def evaluate(beam, num_points, num_classes, model_folder, model_file_stem, data_file_stem):
    """
    Sample invocation:
        python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2023-06-16-16:26:23/full_model models/2023-06-16-16:26:23/weights/cp-043.ckpt 
          voxel_data/Mg22_size512
          
        user changes: models/<date>/weights/<chosen weight>
    """
    # load model
    model = tf.keras.models.load_model(model_folder)

    # load test data
    BATCH_SIZE = 32
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :4]).batch(BATCH_SIZE)
    test_labels = test_ds[:, :, 4]
    
    # make predictions
    predicted_probabilities = model.predict(test_features)
    predictions = np.argmax(predicted_probabilities, axis=2)

    # evaluate results
    model_name = model_file_stem.split("/")[-3]
    ckpt_name = model_file_stem.split("/")[-1]
    print('Mean accuracy: {}'.format(np.mean(test_labels == predictions))) #point-wise accuracy
    plot_events(beam, test_labels, predictions, data_file_stem, model_name, ckpt_name)
    
    # create histogram of percent accuracy
    percent_accuracy = np.mean(test_labels == predictions, axis=1)
    plot_histogram(beam, model_name, percent_accuracy, ckpt_name)
    
    #for analyzing histogram peaks at 0 and 1
    plot_zero_one_bins(beam, test_labels, predictions, data_file_stem, model_name, ckpt_name)

    
if __name__ == '__main__':
    evaluate()