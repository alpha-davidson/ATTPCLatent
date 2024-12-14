import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from benchmark_helper.pointnet import create_pointnet_model
from benchmark_helper.plotting import plot_learning_curves, point_clouds, squared_error_point_clouds
from benchmark_helper.data_norm import data_scale
from sklearn.metrics import r2_score
import os
import pandas as pd
import click
import pickle
from datetime import datetime

# Function to train and predict for each set of data
def train_and_predict(path, file_extension):
    # Train the model
    model = train_event_wise_classification_model(path, file_extension)

    # Load test data for predictions
    X_test = np.load(f'../data_processing/{path}/{file_extension}test_features.npy')
    y_true = np.load(f'../data_processing/{path}/{file_extension}test_labels.npy')

    # Create predictions folder if it doesn't exist
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    preds_path = f"benchmark_predictions/{timestamp}"
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Save predictions
    np.save(f'{preds_path}/{file_extension}predictions.npy', y_pred)
    print(f"Predictions saved to {preds_path}/{file_extension}predictions.npy")

# Function to train the model
def train_event_wise_classification_model(path, file_extension):
    # Load training data for the current iteration
    train_features = np.load(f'../data_processing/{path}/{file_extension}train_features.npy') 
    train_labels = np.load(f'../data_processing/{path}/{file_extension}train_labels.npy')
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_ds = train_ds.batch(batch_size=32, drop_remainder=True)

    # Load validation data
    val_features = np.load(f'../data_processing/{path}/{file_extension}val_features.npy')
    val_labels = np.load(f'../data_processing/{path}/{file_extension}val_labels.npy')
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    val_ds = val_ds.batch(batch_size=32, drop_remainder=True)

    # Model parameters
    num_points = train_features.shape[1]
    num_features = train_features.shape[2]

    # Build and train the classification model
    model = create_pointnet_model(num_points, 
                                  num_features, 
                                  num_classes=len(np.unique(train_labels)),
                                  is_regression=False,
                                  is_pointwise_prediction=False)
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=["sparse_categorical_accuracy"])
        
    # Save the best model during training
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    checkpoint_filepath = f'benchmark_model/{timestamp}/model_checkpoint_{{epoch:03d}}.keras'
    best_model_filepath = f'benchmark_model/{timestamp}/best_model.keras'

    if not os.path.exists(f'benchmark_model/{timestamp}'):
        os.makedirs(f'benchmark_model/{timestamp}')

    # Callbacks for saving models and reducing learning rate on plateau
    periodic_save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, save_freq='epoch')
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_filepath, save_weights_only=False, monitor='loss', mode='min', save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=7, verbose=1)
    
    # Train the model and save the learning curve
    history = model.fit(train_ds, validation_data=val_ds, epochs=150, verbose=1, callbacks=[periodic_save_callback, best_model_callback, reduce_lr])
    plot_learning_curves(history, f'benchmark_model/{timestamp}/learning-curve.png')

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_json_file = f'benchmark_model/{timestamp}/history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    
    # Save and reload the model to ensure it was saved correctly
    model.save(filepath=f'benchmark_model/{timestamp}/best_model.keras')
    reloaded_model = keras.models.load_model(f'benchmark_model/{timestamp}/best_model.keras')

    return reloaded_model

@click.command()
@click.option("--data-folder", help="Data folder")
@click.option("--file-extension", help="File extension")

def train_classification_pointnet_model(data_folder, file_extension):
    if not data_folder or not file_extension:
        print("Usage: python pointnet_classification_and_prediction.py --data-folder <DATA-FOLDER> --file-extension <FILE-EXTENSION>")
        exit(1)

    data_path = f"{data_folder}"
    train_and_predict(data_path, file_extension)
    
if __name__ == '__main__':
    train_classification_pointnet_model()
