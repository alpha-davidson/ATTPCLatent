"""

name: O16_voxel_pipeline.py

description: Takes the raw point cloud data file called 'O16_run160.h5' and creates new files to be placed in 'O16_expt_downstream/voxel_data'. Before running, create a 'voxel_data' folder under the 'O16_expt_downstream' directory and put the 'O16_run160.h5' file into that folder.

designed for data in the following format:
x[0] ,y[1] ,z[2] ,time[3], Amplitude[4]

date created: Jul 22, 2024
date edited: Jul 22, 2024

"""

# imports...

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tqdm
import random
import os.path
import pandas as pd
from sklearn import preprocessing
import os

# user defined functions...

def convert_data(data):
    
    keys = list(data.keys())
    
    # making array of event lengths
    event_lens = np.zeros(len(keys), int)
    for i in range(len(keys)):
        event = keys[i]
        event_lens[i] = len(data[event])
    np.save('../voxel_data/O16_event_lens.npy', event_lens)
    
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
    event_data[56437,0,5] = 56437 # fixing the empty event ('Event 60795,' index 56437)
    np.save('../voxel_data/O16_w_event_keys.npy', event_data)
    
def _filter_point_clouds(data):
    
    # Extract only the x, y, z coordinates, and the charge (columns 0, 1, 2, 4)
    filtered_data = data[..., [0, 1, 2, 4]]
    
    return filtered_data

def filter_data(sample_size, TRACK_CLASS, ISOTOPE):
    
    data = np.load('../voxel_data/' + ISOTOPE + '_w_event_keys.npy')
    event_lens = np.load('../voxel_data/' + ISOTOPE + '_event_lens.npy')
    data[56437,0,-1] = 56437
    
    # Apply the filter function to the data
    filtered_data = _filter_point_clouds(data)
    
    # Apply additional filtering based on the charge value
    filtered_data[:,:,0] = np.where(filtered_data[:,:,3] < 90, 0, filtered_data[:,:,0])
    filtered_data[:,:,1] = np.where(filtered_data[:,:,3] < 90, 0, filtered_data[:,:,1])
    filtered_data[:,:,2] = np.where(filtered_data[:,:,3] < 90, 0, filtered_data[:,:,2])
    filtered_data[:,:,3] = np.where(filtered_data[:,:,3] < 90, 0, filtered_data[:,:,3])
    
    # Determine threshold for minimum number of non-zero points
    min_points_threshold = 70
    
    # Filter out events with too few non-zero points
    filtered_events = []
    valid_event_indices = []
    for i in range(filtered_data.shape[0]):
        if np.count_nonzero(filtered_data[i, :, 0]) >= min_points_threshold:
            filtered_events.append(filtered_data[i])
            valid_event_indices.append(i)
    
    filtered_data = np.array(filtered_events)
    
    np.save('../voxel_data/O16_filtered_data.npy', filtered_data)

def main():
    
    # user inputs
    sample_size = 512 # enter the size to which events will be up/downsampled
    TRACK_CLASS = False
    dimension = 4 # desired dimension of data to be input
    ISOTOPE = 'O16'

    # could try to change to other isotope data in the future
    data = h5py.File('../voxel_data/O16_run160.h5','r')

    # calling the functions
    '''convert_data(data)'''
    filter_data(sample_size, TRACK_CLASS, ISOTOPE)

if __name__ == "__main__":
    
    main()