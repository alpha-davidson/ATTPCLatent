import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from plotting import plot_learning_curve
from pointnet_model import pnet
from datetime import datetime
import click
import os


@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--batch-size', default=32, type=click.INT, help='Batch size for SGD')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.option('--num-epochs', default=10, type=click.INT, help='Number of epochs of training')
@click.argument('file-stem')
def train(num_points, batch_size, num_classes, num_epochs, file_stem):

    # load training data
    train_features_filename = file_stem + '_train_features.npy'
    train_labels_filename = file_stem + '_train_labels.npy'
    train_features = np.load(train_features_filename)
    train_labels = np.load(train_labels_filename)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_ds = train_ds.batch(batch_size, drop_remainder=True)

    # load validation data
    val_features_filename = file_stem + '_val_features.npy'
    val_labels_filename = file_stem + '_val_labels.npy'
    val_features = np.load(val_features_filename)
    val_labels = np.load(val_labels_filename)
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    val_ds = val_ds.batch(batch_size, drop_remainder=True)

    # build and train model
    model = pnet(sem_seg_flag=False, num_points=num_points, num_classes=num_classes)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=["sparse_categorical_accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, verbose=1)
    
    # save model and plot learning curve
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    model_file_path = 'models/{}/weights'.format(timestamp)
    model.save_weights(filepath=model_file_path)
    
    os.makedirs('plots/{}'.format(timestamp))
    plot_file_path = 'plots/{}/learning_curve.png'.format(timestamp)
    plot_learning_curve(history, plot_file_path)
    
    
if __name__ == '__main__':
    train()
