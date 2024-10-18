"""
Script to benchmark PointNet implementation for classification tasks.

Author: RR, WZ, DH
"""
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pointnet.pointnet import create_pointnet_model
from pointnet.plotting import plot_learning_curves, point_clouds, squared_error_point_clouds
from pointnet.data_norm import data_scale
from sklearn.metrics import r2_score
import os
import pandas as pd
import click
import pickle

# If a model is passed in, it is assumed to be located in `model_num`'s corresponding folder
def make_predictions(model_num, path, file_extension, model=None):
    preds_path = f"predictions/{model_num}"
    X_test = np.load(f'{path}test_features.npy')
    y_true = np.load(f'{path}test_labels.npy')

    # If we want to retrain and make predictions
    if not model:   
       model = train_event_wise_classification_model(model_num, path)

    # If evaluating a model for the first time, then create a folder for preds 
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)
    
    # predict number of events from model
    y_pred = model.predict(X_test)
    shape = y_pred.shape

    np.save(preds_path + f'/{file_extension}predictions.npy', y_pred)
    
        
def train_event_wise_classification_model(model_num, path=""):
    # load training data
    train_features = np.load(f'{path}135train_features.npy') # load directly from file to memory,
    train_labels = np.load(f'{path}135train_labels.npy')     # can revert to tf.data.Dataset if necessary
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_ds = train_ds.batch(batch_size=32, drop_remainder=True)

    # load validation data
    val_features = np.load(f'{path}val_features.npy')
    val_labels = np.load(f'{path}val_labels.npy')
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    val_ds = val_ds.batch(batch_size=32, drop_remainder=True)

    num_points = train_features.shape[1]
    num_features = train_features.shape[2]

    # build and train event-wise classification model and plot learning curve
    model = create_pointnet_model(num_points, 
                                  num_features, 
                                  num_classes = len(np.unique(train_labels)),
                                  is_regression=False,
                                  is_pointwise_prediction=False)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=["sparse_categorical_accuracy"])

    # Save best model from training and saves every 10 epochs
    
    if not os.path.exists(f'Mg22_dw_benchmark/{model_num}'):
        os.makedirs(f'Mg22_dw_benchmark/{model_num}')

    checkpoint_filepath = f'Mg22_dw_benchmark/{model_num}/model_checkpoint_{{epoch:03d}}.keras'
    best_model_filepath = f'Mg22_dw_benchmark/{model_num}/best_model.keras'

    # Callback for periodic saving
    periodic_save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq = 'epoch'
    )

    # Callback for saving the best model
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

    
    # Reduce learning rate when model is not improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=7, verbose=1)
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=150, verbose=1, callbacks=[periodic_save_callback, best_model_callback, reduce_lr])
    plot_learning_curves(history, f'Mg22_dw_benchmark/{model_num}/learning-curve.png')

    try:        
        # Save the model history to a file    
        hist_df = pd.DataFrame(history.history)  
        hist_json_file = f"Mg22_dw_benchmark/{model_num}/history.json" 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
    except Exception as e:
        print(e)
    
    # test saving and loading back model
    model.save(filepath=f'Mg22_dw_benchmark/{model_num}/model{model_num}.keras')
    reloaded_model = keras.models.load_model(f'Mg22_dw_benchmark/{model_num}/model{model_num}.keras')
    print('Successfully saved and loaded back model!')


@click.command()
@click.option("--model-num", help="Model number") 
@click.option("--data-folder", help="Data folder")
@click.option("--file-extension", help="File extension")

def train_classification_pointnet_model(model_num, data_folder, file_extension):
    if not model_num or not data_folder or not file_extension:
        print("Usage: python pointnet_classification.py --model-num <MODEL-NUM> --data-folder <DATA-FOLDER> --file-extension <FILE-EXTENSION>")
        exit(1)

    print(f"Model: {model_num} \t\tData Folder: {data_folder} \t\tFile Extension: {file_extension}")

    data_path = f"{data_folder}/{file_extension}"

    train_event_wise_classification_model(model_num, path=data_path)

    #model = keras.models.load_model(f"model_class/{model_num}/best_model.keras")
    #model = keras.models.load_model(f"model_class/{model_num}/model_checkpoint_046.keras")
    #make_predictions(model_num, data_path, file_extension, model)
    
if __name__ == '__main__':
    train_classification_pointnet_model()
