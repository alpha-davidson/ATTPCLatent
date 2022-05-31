import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from plotting import plot_learning_curve
from pointnet_model import pnet
from datetime import datetime
import click


def fix_shape(points, labels):
    # points already have the correct shape
    labels = tf.reshape(labels, (512, 1))
    return points, labels


@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--batch-size', default=32, type=click.INT, help='Batch size for SGD')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.option('--num-epochs', default=10, type=click.INT, help='Number of epochs of training')
@click.argument('train-file-stem')
@click.argument('val-file-stem')
def train(num_points, batch_size, num_classes, num_epochs, train_file_stem, val_file_stem):
    train_points = np.load(train_file_stem + '_points.npy')
    train_labels = np.load(train_file_stem + '_labels.npy')

    val_points = np.load(val_file_stem + '_points.npy')
    val_labels = np.load(val_file_stem + '_labels.npy')

    # need to drop remainder?
    train_ds = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_points, val_labels))
    
    train_ds = train_ds.shuffle(len(train_ds)).batch(batch_size)
    val_ds = val_ds.shuffle(len(val_ds)).batch(batch_size)
    
    model = pnet(sem_seg_flag=False, num_points=num_points, num_classes=num_classes)
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=["sparse_categorical_accuracy"])
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, verbose = 1)
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    model_file_path = 'models/{}/weights'.format(timestamp)
    model.save_weights(filepath=model_file_path)
    
    plot_file_path = 'plots/{}/learning_curve.png'.format(timestamp)
    plot_learning_curve(history, plot_file_path)
    
    
if __name__ == '__main__':
    train()
