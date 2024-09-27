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

def get_latest_checkpoint(checkpoint_dir):
    """
    Additional function to get the latest checkpoint file
    """
    try:
        paths = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)]
        return max(paths, key=os.path.getctime)
    except FileNotFoundError as error:
        print(f"No checkpoint found in {checkpoint_dir}!!!!!")
        raise error
    except ValueError as error:
        print(f"No valid checkpoint found in {checkpoint_dir}!!!!!")
        raise error

@click.command()
@click.option('--element', default='O16', type=click.STRING, help='The element to train on (e.g. O16, Mg22, C16)')
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--batch-size', default=32, type=click.INT, help='Batch size for SGD')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.option('--num-epochs', default=10, type=click.INT, help='Number of epochs of training')
@click.option('--fine-tune', default=None, type=click.STRING, help='Path to a specific checkpoint file for fine-tuning')
@click.argument('file-stem')
def train(element, num_points, batch_size, num_classes, num_epochs, fine_tune, file_stem):
    """
    Sample invocation:
        python3 pretrain_on_jigsaw_events.py --element O16 --num-classes 24 --num-epochs 50 O16_pretrain/voxel_data/O16_size512
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
    
    # Build model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)

    # Load specific checkpoint if fine-tuning
    if fine_tune:
        model.load_weights(fine_tune)
        print(f"Loaded weights from {fine_tune} for fine-tuning.")
    
    # build and fit model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)

    # save model and plot learning curve
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    checkpoint_path = f"{element}_models/{timestamp}/weights/cp"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_path = checkpoint_path + "-{epoch:03d}.ckpt"
        
    # callback checkpoint to save model every [save_freq] epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only = True,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = False,
        save_freq = 'epoch')
    
    # Ensure directory exists before saving initial weights
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_weights(checkpoint_path.format(epoch=0))
    
    #reduce LR checkpoint to adjust LR upon no improvement
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        mode='auto',
        verbose=1,
        min_delta=0.001,
        min_lr=0.00001)
    
    # build model and fit model
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.005), 
                  metrics=["sparse_categorical_accuracy", "accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, 
                        callbacks=[checkpoint_callback, reduce_lr], verbose=1)

    model_path = f"{element}_models/{timestamp}/full_model"
    model.save(model_path)
    
    os.makedirs('{element}_plots/{}'.format(timestamp))
    plot_file_path = '{element}_plots/{}/learning_curve.png'.format(timestamp)
    plot_learning_curve(history, plot_file_path)
    
if __name__ == '__main__':
    train()