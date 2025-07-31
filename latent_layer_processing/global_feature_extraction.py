import os
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..pointnet import create_pointnet_model

def extraction(data_file_stem, BATCH_SIZE):
    """
    Loads 3D event data from a .npy file and converts it into a batched TensorFlow dataset.

    Parameters:
        data_file_stem : Path to the .npy file without extension.
        BATCH_SIZE : Number of samples per batch.

    Returns:
        tf.data.Dataset: Batched dataset containing only the first 4 features per point.
    """
    ds = np.load(f'{data_file_stem}.npy')
    features = tf.data.Dataset.from_tensor_slices(ds[:, :, :4]).batch(BATCH_SIZE)
    return features

def global_features(model, data_file_stem, BATCH_SIZE):
    """
    Generates global feature representations from a pretrained model's latent layer.

    Parameters:
        model : Model truncated at the latent layer.
        data_file_stem : Path to the .npy data file (no extension).
        BATCH_SIZE : Batch size for prediction.

    Returns:
        np.ndarray: Array of global feature vectors for each event.
    """
    features = extraction(data_file_stem, BATCH_SIZE)
    global_features = model.predict(features)
    return global_features

def test_train_split(ISOTOPE, global_features):
    """
    Randomly splits the global features into training and testing sets (80/20) and saves them.

    Parameters:
        ISOTOPE : Isotope identifier used to name output directory and files.
        global_features : The extracted global features.
    """
    name = ISOTOPE + "global_features"
    all_events = global_features

    # Shuffle event indices
    rand_shuffle = np.random.choice(len(all_events), len(all_events), replace=False)

    # 20% for test
    test_split = int(len(all_events) * 0.2)

    test_data = all_events[rand_shuffle[:test_split], :]
    train_data = all_events[rand_shuffle[test_split:], :]

    folder_path = f'./global_features/{ISOTOPE}'
    os.makedirs(folder_path, exist_ok=True)

    print(f"Test shape: {test_data.shape}, Train shape: {train_data.shape}")
    np.save(f'{folder_path}/test', test_data)
    np.save(f'{folder_path}/train', train_data)

@click.command()
@click.argument('model-folder')
@click.argument('data-file')
@click.argument('file-dir')
def extract_global_features(model_folder, data_file, file_dir):
    """
    Extracts global features from a trained model's latent layer and saves them to disk.

    Parameters:
        model_folder : Path to the trained Keras model directory.
        data_file : Path (without .npy) to the voxelized 3D input data.
        file_dir : Path where the resulting global features .npy file will be saved.

    Example:
        PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_extraction 
        ../training/O16_models/2025-06-16-14:56:34/full_model 
        ../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512 
        global_features/O16_3track.npy
    """
    # Load trained model
    model = tf.keras.models.load_model(model_folder)
    BATCH_SIZE = 32

    # Define model up to latent layer
    global_feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)

    # Extract and save global features
    data_global_features = global_features(global_feature_extractor, data_file, BATCH_SIZE)
    print(f"Global Features Shape: {data_global_features.shape}")

    os.makedirs('./global_features', exist_ok=True)
    np.save(file_dir, data_global_features)

if __name__ == '__main__':
    extract_global_features()
