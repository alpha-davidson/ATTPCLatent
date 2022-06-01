import plotting
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from pointnet_model import pnet
import click

   
@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('cm-file-path')
@click.argument('model-file-stem')
@click.argument('data-file-stem')
def evaluate(num_points, num_classes, cm_file_path, model_file_stem, data_file_stem):
    """
    Example invocation:
        python3 evaluate_event_classification_model.py --num-classes 6 plots/2022-06-01-14:54:51/cm.png \
          models/2022-06-01-14:54:51/weights data/Mg22_size512_convertXYZ_test
    """
    # build model
    model = pnet(sem_seg_flag=False, num_points=num_points, num_classes=num_classes)
    model.load_weights(model_file_stem)

    # load testing data
    BATCH_SIZE = 16
    test_features_filename = data_file_stem + '_features.npy'
    test_labels_filename = data_file_stem + '_labels.npy'
    test_features = np.load(test_features_filename)
    test_labels = np.load(test_labels_filename)
    test_ds = tf.data.Dataset.from_tensor_slices(test_features)
    test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    trimmed_length = len(test_labels) - (len(test_labels) % BATCH_SIZE)
    test_labels = test_labels[:trimmed_length]

    # make predictions
    predicted_probabilities = model.predict(test_ds)
    predictions = np.argmax(predicted_probabilities, axis=1)
    
    # output results
    class_names = list(map(str, range(num_classes)))
    print(classification_report(test_labels, predictions, target_names=class_names))
    plotting.plot_confusion_matrix(test_labels, predictions, class_names, cm_file_path)


if __name__ == "__main__":
    evaluate()