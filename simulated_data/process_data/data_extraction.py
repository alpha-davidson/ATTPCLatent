"""

name: O16_voxel_pipeline.py

description: Takes the raw point cloud data file called 'O16_run160.h5' and creates new files to be placed in 'O16_expt_downstream/voxel_data'. Before running, create a 'voxel_data' folder under the 'O16_expt_downstream' directory and put the 'O16_run160.h5' file into that folder.

designed for data in the following format:
x[0] ,y[1] ,z[2] ,time[3], Amplitude[4]

date created: Jul 22, 2024
date edited: Jul 24, 2024

"""

# imports...

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tqdm
import click
import random
import os.path
import pandas as pd
from sklearn import preprocessing
import os
import json

# user defined functions...

def convert_data(ISOTOPE, data, folder_path):
    """
    Takes in point cloud data as an .h5 file and converts it into a numpy array.
    
    Parameters
    ----------
    data : h5
        Raw point cloud data.
    
    Returns
    ----------
    None.
    """
    
    keys = list(data.keys())
    
    # making array of event lengths
    event_lens = np.zeros(len(keys), int)
    for i in range(len(keys)):
        event = keys[i]
        event_lens[i] = len(data[event])
    np.save(f'{ISOTOPE}/voxel_data/{folder_path}/{ISOTOPE}_event_lens.npy', event_lens)
    
    # making a numpy array of data
    event_data = np.zeros((len(keys), np.max(event_lens), 6), float)
    for n in tqdm.tqdm(range(len(keys))):
        name = keys[n]
        event = data[name]
        ev_len = len(event)
        for i,e in enumerate(event):
            instant = np.array(list(e))
            event_data[n,i,:5] = instant[:5]
            event_data[n,i,5] = float(n) # storing the event index

    np.save(f'{ISOTOPE}/voxel_data/{folder_path}/{ISOTOPE}_w_event_keys.npy', event_data)
    
def _filter_point_clouds(data):
    
    # Extract only the x, y, z coordinates, and the charge (columns 0, 1, 2, 4)
    filtered_data = data[..., [0, 1, 2, 4]]
    
    return filtered_data

def filter_data(ISOTOPE, min_points_threshold, min_charge_threshold, folder_path):
    """
    Filters out all events that don't have at least 70 points.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. O16).
    
    min_points_threshold : int
        Minimum points to be safe.

    min_charge_threshold : int
        Minimum charge value to be safe.
    
    Returns
    ----------
    None.
    """
    
    data = np.load(f'{ISOTOPE}/voxel_data/{folder_path}/' + ISOTOPE + '_w_event_keys.npy')
    event_lens = np.load(f'{ISOTOPE}/voxel_data/{folder_path}/' + ISOTOPE + '_event_lens.npy')

    # Apply the filter function to the data
    filtered_data = _filter_point_clouds(data)

    # Apply additional filtering based on the charge value
    filtered_data[:,:,0] = np.where(filtered_data[:,:,3] < min_charge_threshold, 0, filtered_data[:,:,0])
    filtered_data[:,:,1] = np.where(filtered_data[:,:,3] < min_charge_threshold, 0, filtered_data[:,:,1])
    filtered_data[:,:,2] = np.where(filtered_data[:,:,3] < min_charge_threshold, 0, filtered_data[:,:,2])
    filtered_data[:,:,3] = np.where(filtered_data[:,:,3] < min_charge_threshold, 0, filtered_data[:,:,3])

    # Filter out events with too few non-zero points
    filtered_events = []
    valid_event_indices = []

    for i in range(filtered_data.shape[0]):
        if np.count_nonzero(filtered_data[i, :, 0]) >= min_points_threshold:
            filtered_events.append(filtered_data[i])
            valid_event_indices.append(i)
    
    filtered_data = np.array(filtered_events)
    valid_event_indices = np.array(valid_event_indices)
    
    np.save(f'{ISOTOPE}/voxel_data/{folder_path}/{ISOTOPE}_filtered_data', filtered_data)
    np.save(f'{ISOTOPE}/voxel_data/{folder_path}/{ISOTOPE}_valid_event_indices', valid_event_indices)

def random_sample(ISOTOPE, sample_size, dimension, folder_path):
    """
    Each event might have any number of points over the min_point_threshold. This function condenses/expands each event to contain 512 instances.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. O16).

    sample_size : int
        The number of instances you want each event to become.

    dimension : int
        Desired dimension of data for input.
    
    Returns
    ----------
    None.
    """

    data = np.load(f'{ISOTOPE}/voxel_data/{folder_path}/' + ISOTOPE + '_w_event_keys.npy')
    filtered_data = np.load(f'{ISOTOPE}/voxel_data/{folder_path}/{ISOTOPE}_filtered_data.npy')
    valid_event_indices = np.load(f'{ISOTOPE}/voxel_data/{folder_path}/{ISOTOPE}_valid_event_indices.npy')
    
    new_array_name = ISOTOPE + '_size' + str(sample_size) + '_sampled'
    new_data = np.zeros((filtered_data.shape[0], sample_size, 6), float)
    # Using tqdm with enumerate for progress indication
    for idx, original_idx in tqdm.tqdm(enumerate(valid_event_indices), total=len(valid_event_indices)):
        # Filter out zero values using idx instead of original_idx
        non_zero_points = filtered_data[idx][filtered_data[idx, :, 0] != 0]
        non_zero_len = non_zero_points.shape[0]
        
        if non_zero_len > sample_size:
            random_points = np.random.choice(non_zero_len, sample_size, replace=False)  # choosing the random instances to sample
            for count, r in enumerate(random_points):
                new_data[idx, count, :3] = non_zero_points[r, :3]  # Only use the filtered x, y, z
                new_data[idx, count, 3] = data[original_idx, r, 3]  # adding time from original data
                new_data[idx, count, 4] = non_zero_points[r, 3]  # use amplitude (charge) from filtered data
                new_data[idx, count, 5:] = data[original_idx, r, 5:]  # Add the remaining columns (event index) from the original data
        else:
            new_data[idx, :non_zero_len, :3] = non_zero_points[:, :3]  # Only use the filtered x, y, z
            new_data[idx, :non_zero_len, 3] = data[original_idx, :non_zero_len, 3]  # adding time from original data
            new_data[idx, :non_zero_len, 4] = non_zero_points[:, 3]  # use amplitude (charge) from filtered data
            new_data[idx, :non_zero_len, 5:] = data[original_idx, :non_zero_len, 5:]  # Add the remaining columns (event index) from the original data
            need = sample_size - non_zero_len
            random_points = np.random.choice(non_zero_len, need, replace=True if need > non_zero_len else False)
            for count, r in enumerate(random_points, start=non_zero_len):
                new_data[idx, count, :3] = non_zero_points[r, :3]  # Only use the filtered x, y, z
                new_data[idx, count, 3] = data[original_idx, r, 3]  # adding time from original data
                new_data[idx, count, 4] = non_zero_points[r, 3]  # use amplitude (charge) from filtered data
                new_data[idx, count, 5:] = data[original_idx, r, 5:]  # Add the remaining columns (event index) from the original data
        new_data[idx, 0, 5] = data[original_idx, 0, 5]  # saving the event index
    
    # Verify if there are still any zero values in new_data
    print("Final check of new_data for zeros")
    print("Number of zero values in x, y, z, charge columns:", np.count_nonzero(new_data[:, :, :4] == 0))
    print("Indices of zero values:", np.where(new_data[:, :, :4] == 0))
    
    # If there are still zeros, print a few samples where zeros are present
    zero_indices = np.where(new_data[:, :, :4] == 0)
    if len(zero_indices[0]) > 0:
        for i in range(min(5, len(zero_indices[0]))):
            print(f"Zero found at index {zero_indices[0][i]}, event {zero_indices[1][i]}, column {zero_indices[2][i]}")
    
    assert np.all(new_data[:, :, :4] != 0), 'new_data contains zero values in the x, y, z, or charge columns'
    
    np.save(f'{ISOTOPE}/voxel_data/{folder_path}/' + new_array_name, new_data)
    
    assert new_data.shape == (filtered_data.shape[0], sample_size, 6), 'Array has incorrect shape'
    assert len(np.unique(new_data[:, :, 5])) == filtered_data.shape[0], 'Array has incorrect number of events'

    size = len(new_data)
    dataset = np.zeros((size, sample_size, dimension + 3), float)
    count = 0

    for i in range(size):
        dataset[count,:,:3] = new_data[count,:,:3] # x, y, z
        if dimension == 4: 
            dataset[count,:,3] = new_data[count,:,4] # charge
        dataset[count,0,-3] = new_data[count,0,5] # event index
        # [count,0,-2] -- number of tracks, leave as 0
        dataset[count,0,-1] = np.count_nonzero(new_data[count,:,0])# length of event 
        count += 1
    
    np.save(f'{ISOTOPE}/voxel_data/{folder_path}/' + ISOTOPE + '_size' + str(sample_size), dataset)

@click.command()
@click.option('--beam', default='O16', type=click.STRING, help='The beam to train on (e.g. O16, Mg22)')
@click.argument('file')

def extract_data(beam, file):
    # user inputs
    sample_size = 512 # enter the size to which events will be up/downsampled
    TRACK_CLASS = False
    dimension = 4 # desired dimension of data to be input
    ISOTOPE = beam
    
    min_points_threshold = 30 # Determine threshold for minimum number of non-zero points (to be used in the filter_data function).  
    min_charge_threshold = 5 # ^ same but for charge value

    K_x = 2
    K_y = 2
    K_z = 6

    # open a file whose data from which the data will be extracted
    data = h5py.File(f'{ISOTOPE}/voxel_data/{file}.h5','r')

    # create a folder for data
    folder_path = f'{ISOTOPE}/voxel_data/{file}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # calling the functions
    convert_data(ISOTOPE, data, file)
    filter_data(ISOTOPE, min_points_threshold, min_charge_threshold, file)
    random_sample(ISOTOPE, sample_size, dimension, file)

if __name__ == "__main__":
    extract_data()