import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from pointnet_model import pnet
from plotting import plot_events, plot_histogram


@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('model-file-stem')
@click.argument('data-file-stem')
def evaluate(num_points, num_classes, model_file_stem, data_file_stem):
    """
    Sample invocation:
        python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2022-10-17-14:41:20/weights \
          voxel_data/Mg22_size512
    """
    # build model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)
    model.load_weights(model_file_stem)

    # load test data
    BATCH_SIZE = 32
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :3]).batch(BATCH_SIZE)
    test_labels = test_ds[:, :, 3]
    
    # make predictions
    predicted_probabilities = model.predict(test_features)
    predictions = np.argmax(predicted_probabilities, axis=2)

    # evaluate results
    model_name = model_file_stem.split("/")[-3]
    print('Mean accuracy: {}'.format(np.mean(test_labels == predictions))) #point-wise accuracy
    plot_events(test_labels, predictions, data_file_stem, model_name)
    
    # create histogram of percent accuracy
    model_name = model_file_stem.split("/")[-3]
    percent_accuracy = np.mean(test_labels == predictions, axis=1)
    plot_histogram(model_name, percent_accuracy)
    

    
if __name__ == '__main__':
    evaluate()
