import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import sys
import cnn_functions as cnn
from models.pointnet_model import pnet

tf.random.set_seed(1234)

def fix_shape(points, labels):
    # points already have the correct shape
    labels = tf.reshape(labels, (512, 1))
    return points, labels

def train():
    sem_seg_flag = config['sem_seg_flag']
    voxel_shuffle_flag = config['voxel_shuffle_flag']
    
    if voxel_shuffle_flag:
        train_ds = np.load(config['train_points3'])
        val_ds = np.load(config['val_points3'])

        train_features = train_ds[:, :, :3]
        train_labels = train_ds[:, :, 3]
        train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).map(fix_shape)

        val_features = val_ds[:, :, :3]
        val_labels = val_ds[:, :, 3]
        val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).map(fix_shape)
    else:
        if sem_seg_flag:
            train_ds = np.load(config['train_points2'])
            val_ds = np.load(config['val_points2'])

            train_features = train_ds[:, :, :3]
            train_labels = train_ds[:, :, 3]
            train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).map(fix_shape)

            val_features = val_ds[:, :, :3]
            val_labels = val_ds[:, :, 3]
            val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).map(fix_shape)
        else:
            train_points = np.load(config['train_points1'])
            train_labels = np.load(config['train_labels1'])

            val_points = np.load(config['val_points1'])
            val_labels = np.load(config['val_labels1'])

            train_ds = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
            val_ds = tf.data.Dataset.from_tensor_slices((val_points, val_labels))
    
    train_ds = train_ds.shuffle(len(train_ds)).batch(config['batch_size'])
    val_ds = val_ds.shuffle(len(val_ds)).batch(config['batch_size'])
    
    model = pnet(config['sem_seg_flag'], config['num_points'], int(config['num_classes']))
    
    model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    metrics=["sparse_categorical_accuracy"],
    )
    
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose = 1
    )
    
    model.save_weights(filepath = './logs/pnet_1/model/weights')
    cnn.plot_learning_curve(history)
    
    
if __name__ == '__main__':
    
    config = {
    'train_points1' : 'data/Mg22_size512train_convertXYZ.npy',
    'train_labels1' : 'data/Mg22_size512_track_labels_train_convertXYZ.npy',
    'val_points1' : 'data/Mg22_size512val_convertXYZ.npy',
    'val_labels1' : 'data/Mg22_size512_track_labels_val_convertXYZ.npy',
    'train_points2' : 'data2/Mg22_size512train_convertXYZ.npy',
    'val_points2' : 'data2/Mg22_size512val_convertXYZ.npy',
    'train_points3' : 'data3/Mg22_size512train.npy',
    'val_points3' : 'data3/Mg22_size512val.npy',
    'num_points' : 512,
    'batch_size' : 32,
    'sem_seg_flag' : True, #sys.argv[1],
    'num_classes' : sys.argv[2],
    'voxel_shuffle_flag' : sys.argv[3]
    }
    train()