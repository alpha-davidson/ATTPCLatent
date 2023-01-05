import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from plotting import plot_learning_curve
from pointnet_model import pnet
from datetime import datetime
import click
import os


def fix_shape(points, labels):
    # points already have the correct shape, so only reshape labels
    labels = tf.reshape(labels, (512, 1))
    return points, labels


@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--batch-size', default=32, type=click.INT, help='Batch size for SGD')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.option('--num-epochs', default=10, type=click.INT, help='Number of epochs of training')
@click.argument('file-stem')
def train(num_points, batch_size, num_classes, num_epochs, file_stem):
    """
    Sample invocation:
        python3 pretrain_on_jigsaw_events.py --num-classes 27 --num-epochs 50 voxel_data/Mg22_size512
    """
    # load data
    train_ds = np.load('{}{}'.format(file_stem, 'train.npy'))
    val_ds = np.load('{}{}'.format(file_stem, 'val.npy'))
    train_features = train_ds[:, :, :3]
    train_labels = train_ds[:, :, 3].astype(int)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).map(fix_shape)

    val_features = val_ds[:, :, :3]
    val_labels = val_ds[:, :, 3].astype(int)
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).map(fix_shape)
    
    train_ds = train_ds.shuffle(len(train_ds)).batch(batch_size)
    val_ds = val_ds.shuffle(len(val_ds)).batch(batch_size)
    
    # build and fit model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0005), # 0.0005 before alteration
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