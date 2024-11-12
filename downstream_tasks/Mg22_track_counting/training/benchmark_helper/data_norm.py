"""
Script to properly scale/normalize data
Authors: DuBose Tuller, Brian Peacock
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import pickle


# Scales a *single* features according to the specified sclaler
def scale_feature(x_train, x_test, scaler):
    scaled_x_train = scaler.fit_transform(x_train.reshape(-1,1))
    scaled_x_test = scaler.transform(x_test.reshape(-1,1))
    return scaled_x_train, scaled_x_test


"""
Converts the raw feature and label numpy files into train/test split data files
    that have been normalized. x,y,z hard-coded to match the AT-TPC,
    while amplitude and labels use a Quantile Transformer, since the shape of its
    distribution is not close to normal.

Parameters:
    path: relative path to the folder containing the raw data files
    prefix: what all current/newly generated files with start with.
        The function expects that there will be two files accessed by
            <path>final_f.npy, and
            <path>final_l.npy

            which can be generated from the `data_pipeline` directory
"""
def data_scale(path = "../../voxel_data"):
    feature_data = np.load(f"{path}final_f.npy")
    label_data = np.load(f"{path}final_l.npy")

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2, random_state=42)

    # write unscaled y_test to a file to directly compare with predictions
    np.save(f"{path}test_ground_truth.npy", y_test)

    points_per_event = feature_data.shape[1]
    num_features = feature_data.shape[2]
    num_labels = label_data.shape[2]
    train_events = X_train.shape[0]
    test_events = X_test.shape[0]

    scaler_amp = QuantileTransformer(output_distribution='normal', random_state=42)
    scaler_label = QuantileTransformer(output_distribution='normal', random_state=42)

    # Scale x,y,z in place according to the physical dimensions of the detector (with some padding on x,y):
    # x,y: (-275, 275)
    X_train[:, :, 0:2] = (X_train[:, :, 0:2]/550)+0.5 
    X_test[:, :, 0:2] = (X_test[:, :, 0:2]/550)+0.5

    # z: (0, 1000)
    X_train[:, :, 2] /= 1000
    X_test[:, :, 2] /= 1000                    

    scaled_amp_train, scaled_amp_test = scale_feature(X_train[:,:,3], X_test[:,:,3], scaler_amp)
    y_train, y_test = scale_feature(y_train, y_test, scaler_label) # no need to recombine

    X_train[:,:,3] = scaled_amp_train.reshape((train_events, points_per_event))
    X_test[:,:,3]  = scaled_amp_test.reshape((test_events, points_per_event))
    
    y_train = y_train.reshape((train_events, points_per_event, num_labels))
    y_test = y_test.reshape((test_events, points_per_event, num_labels))

    # write scaled data to files for model training ONLY
    np.save(f"{path}train_features.npy", X_train)
    np.save(f"{path}test_features.npy", X_test)  
    np.save(f"{path}train_labels.npy", y_train)  
    np.save(f"{path}test_labels.npy", y_test)    

    # write labels scalers to file
    with open(f"{path}QuantileTransformer", "wb") as p:
        pickle.dump(scaler_label, p)

    # ensure the scaler(s) can be unpickled
    with open(f"{path}QuantileTransformer", "rb") as p:
        assert pickle.load(p) is not None

    
    
