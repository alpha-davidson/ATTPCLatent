"""

name: C16+O16_voxel_pipeline.py

description: Takes the raw point cloud data file called 'C16_run160.h5' and creates new files to be placed in 'C16_expt_downstream/voxel_data'. Before running, create a 'voxel_data' folder under the 'C16_expt_downstream' directory and put the 'C16_run160.h5' file into that folder.

designed for data in the following format:
# x[0] ,y[1] ,z[2] ,Amplitude[3], id[4]


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
import json

# user defined functions...

def load_data():

    C16_data = np.load('../../C16/voxel_data/' + 'C16' + '_w_event_keys.npy', allow_pickle=True)
    C16_event_lens = np.load('../../C16/voxel_data/' + 'C16' + '_event_lens.npy', allow_pickle=True)

    O16_data = np.load('../../O16/voxel_data/' + 'O16' + '_w_event_keys.npy')
    # cut O16 data's time dimension
    O16_data[:,:,3] = O16_data[:,:,4]
    O16_data[:,:,4] = O16_data[:,:,5]
    O16_data = O16_data[:, :, :5]
    O16_event_lens = np.load('../../O16/voxel_data/' + 'O16' + '_event_lens.npy')
    O16_data[56437,0,-1] = 56437
    
    return C16_data, C16_event_lens, O16_data, O16_event_lens


def filter_point_clouds_C16(C16_data):
    C16_filtered_data = C16_data
    
    # Apply additional filtering based on the charge value
    # filtered_data[:,:,0] = np.where(filtered_data[:,:,3] < 70, 0, filtered_data[:,:,0])
    # filtered_data[:,:,1] = np.where(filtered_data[:,:,3] < 70, 0, filtered_data[:,:,1])
    # filtered_data[:,:,2] = np.where(filtered_data[:,:,3] < 70, 0, filtered_data[:,:,2])
    # filtered_data[:,:,3] = np.where(filtered_data[:,:,3] < 70, 0, filtered_data[:,:,3])
    
    #Apply additional filtering based on the z-axis
    C16_filtered_data[:,:,0] = np.where(C16_filtered_data[:,:,2] < 0, 0, C16_filtered_data[:,:,0])
    C16_filtered_data[:,:,1] = np.where(C16_filtered_data[:,:,2] < 0, 0, C16_filtered_data[:,:,1])
    C16_filtered_data[:,:,2] = np.where(C16_filtered_data[:,:,2] < 0, 0, C16_filtered_data[:,:,2])
    C16_filtered_data[:,:,3] = np.where(C16_filtered_data[:,:,2] < 0, 0, C16_filtered_data[:,:,3])
    
    # Determine threshold for minimum number of non-zero points
    min_points_threshold = 60
    
    # Filter out events with too few non-zero points
    C16_filtered_events = []
    C16_valid_event_indices = []
    for i in range(C16_filtered_data.shape[0]):
        if np.count_nonzero(C16_filtered_data[i, :, 0]) >= min_points_threshold:
            C16_filtered_events.append(C16_filtered_data[i])
            C16_valid_event_indices.append(i)
    
    C16_filtered_data = np.array(C16_filtered_events)
    
    # Print the final filtered data for verification
    print('Final Filtered Shape:', C16_filtered_data.shape)
    print(len(C16_valid_event_indices))
    return C16_valid_event_indices, C16_filtered_data

def filter_point_clouds_O16(O16_data):    
    O16_filtered_data = O16_data
    
    # Apply additional filtering based on the charge value
    O16_filtered_data[:,:,0] = np.where(O16_filtered_data[:,:,3] < 90, 0, O16_filtered_data[:,:,0])
    O16_filtered_data[:,:,1] = np.where(O16_filtered_data[:,:,3] < 90, 0, O16_filtered_data[:,:,1])
    O16_filtered_data[:,:,2] = np.where(O16_filtered_data[:,:,3] < 90, 0, O16_filtered_data[:,:,2])
    O16_filtered_data[:,:,3] = np.where(O16_filtered_data[:,:,3] < 90, 0, O16_filtered_data[:,:,3])
    
    # Determine threshold for minimum number of non-zero points
    min_points_threshold = 70
    
    # Filter out events with too few non-zero points
    O16_filtered_events = []
    O16_valid_event_indices = []
    for i in range(O16_filtered_data.shape[0]):
        if np.count_nonzero(O16_filtered_data[i, :, 0]) >= min_points_threshold:
            O16_filtered_events.append(O16_filtered_data[i])
            O16_valid_event_indices.append(i)
    
    O16_filtered_data = np.array(O16_filtered_events)
    
    # Print the final filtered data for verification
    print('Final Filtered Shape:', O16_filtered_data.shape)
    print(len(O16_valid_event_indices))
    return O16_valid_event_indices, O16_filtered_data

def combine_data(C16_filtered_data, C16_valid_event_indices, C16_event_lens, O16_filtered_data, O16_valid_event_indices, O16_event_lens):
    
    diff = np.max(O16_event_lens) - np.max(C16_event_lens)
    C16_filtered_data_padded = np.pad(C16_filtered_data, ((0, 0), (0, diff), (0, 0)), 'constant', constant_values=0)
    data = np.concatenate((C16_filtered_data_padded, O16_filtered_data), axis=0)
    event_lens = C16_valid_event_indices + O16_valid_event_indices
    print(data.shape)
    print(len(event_lens))
    assert data.shape == (len(event_lens), np.max(O16_event_lens), 5), 'Array has incorrect shape'

    np.save('../voxel_data/C16+O16_event_lens.npy', event_lens)
    np.save('../voxel_data/C16+O16_data.npy', data)

    
def random_sample(ISOTOPE, sample_size, dimension):
    """
    Each event might have any number of points over the min_point_threshold. This function condenses/expands each event to contain 512 instances.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. C16).

    sample_size : int
        The number of instances you want each event to become.

    dimension : int
        Desired dimension of data for input.
    
    Returns
    ----------
    None.
    """

    event_lens = np.load('../voxel_data/C16+O16_event_lens.npy')
    data = np.load('../voxel_data/C16+O16_data.npy')
        
    new_array_name = ISOTOPE + '_size' + str(sample_size) + '_sampled'
    new_data = np.zeros((data.shape[0], sample_size, 4), float)
    # Using tqdm with enumerate for progress indication
    for idx, original_idx in tqdm.tqdm(enumerate(event_lens), total=len(event_lens)):
        # Filter out zero values using idx instead of original_idx
        non_zero_points = data[idx][data[idx, :, 0] != 0]
        non_zero_len = non_zero_points.shape[0]
    
        if non_zero_len >= sample_size:
            random_points = np.random.choice(non_zero_len, sample_size, replace=False)  # choosing the random instances to sample
            for count, r in enumerate(random_points):
                new_data[idx, count, :] = non_zero_points[r, :4]  # Only use the filtered x, y, z, and charge
                #new_data[idx, count, 4] = data[original_idx, r, 4]  # Add event index from the original data
    
        else:
            new_data[idx, :non_zero_len, :] = non_zero_points[:, :4]  # Only use the filtered x, y, z, and charge
            #new_data[idx, :non_zero_len, 4] = data[original_idx, :non_zero_len, 4]  # Add event index from the original data
            need = sample_size - non_zero_len
            random_points = np.random.choice(non_zero_len, need, replace=True if need > non_zero_len else False)
            for count, r in enumerate(random_points, start=non_zero_len):
                new_data[idx, count, :] = non_zero_points[r, :4]  # Only use the filtered x, y, z, and charge
                #new_data[idx, count, 4] = data[original_idx, r, 4]  # Add event index from the original data
        #new_data[idx, 0, 4] = data[original_idx, 0, 4]  # saving the event index        
    
    # Verify if there are still any zero values in new_data
    print("Final check of new_data for zeros")
    print("Number of zero values in x, y, z, charge columns:", np.count_nonzero(new_data[:, :, :4] == 0))
    print("Indices of zero values:", np.where(new_data[:, :, :] == 0))
    
    # If there are still zeros, print a few samples where zeros are present
    zero_indices = np.where(new_data[:, :, :] == 0)
    if len(zero_indices[0]) > 0:
        for i in range(min(5, len(zero_indices[0]))):
            print(f"Zero found at index {zero_indices[0][i]}, event {zero_indices[1][i]}, column {zero_indices[2][i]}")
    
    assert np.all(new_data[:, :, :] != 0), 'new_data contains zero values in the x, y, z, or charge columns'
    
    np.save('../voxel_data/' + new_array_name, new_data)
    
    assert new_data.shape == (data.shape[0], sample_size, 4), 'Array has incorrect shape'
    #assert len(np.unique(new_data[:, :, 4])) == data.shape[0], 'Array has incorrect number of events'

    size = len(new_data)
    dataset = np.zeros((size, sample_size, dimension + 2), float)
    count = 0
    
    for i in range(size):
        dataset[count,:,:4] = new_data[count,:,:] # x, y, z, charge
        # dataset[count,0,4] is zero
        dataset[count,0,5] = np.count_nonzero(new_data[count,:,0])# length of event 
        count += 1
    
    np.save('../voxel_data/' + ISOTOPE + '_size' + str(sample_size), dataset)
    print(len(dataset))

def scale_and_split(ISOTOPE, sample_size):
    """
    Scale the data to fit in a 1x1x1 cube.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. C16).

    sample_size : int
        The number of instances you want each event to become.
    
    Returns
    ----------
    None.
    """

    dataset = np.load('../voxel_data/' + ISOTOPE + '_size' + str(sample_size) + '.npy')
    # values correspond to the x,y,z,charge index
    values = [0,1,2,3] 
    means_and_stds = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    dataset[:,:,3] = np.log(dataset[:,:,3]) # log scale charge
    # standard scaling 
    for n in values:
        mean = np.mean(dataset[:,:,n])
        std = np.std(dataset[:,:,n])
        means_and_stds.append([mean,std])
        dataset[:,:,n] = (dataset[:,:,n] - mean) / std
        
    dataset[:,0,-1] = np.log(dataset[:,0,-1])
    dataset[:,0,-1] = min_max_scaler.fit_transform(dataset[:,0,-1].reshape(-1, 1)).reshape(1,-1)
    
    assert np.sum(np.isnan(dataset)) == 0, 'NaNs in dataset'
    assert np.sum(np.isinf(dataset)) == 0, 'Infinities in dataset'


    rand_shuffle = np.random.choice(len(dataset), len(dataset), replace = False)
    name = ISOTOPE + '_size' + str(sample_size)
    
    # 20-20 marking for test and validation
    test_split = int(len(dataset) * .2)
    val_split = int(len(dataset) * .4)
    
    test = dataset[rand_shuffle[:test_split],:,:]
    val = dataset[rand_shuffle[test_split:val_split],:,:]
    train = dataset[rand_shuffle[val_split:],:,:]
    print(len(dataset))
    print(test.shape, val.shape, train.shape)
    
    os.makedirs('../data_splits/', exist_ok=True)
    np.save('../data_splits/' + ISOTOPE + '_size' + str(sample_size)+'_test', test)
    np.save('../data_splits/' + ISOTOPE + '_size' + str(sample_size)+'_val', val)
    np.save('../data_splits/' + ISOTOPE + '_size' + str(sample_size)+'_train', train)
    assert len(np.unique(np.isnan(train[:,:,4]))) == 1, 'NaNs in dataset'
    assert len(np.unique(np.isnan(val[:,:,4]))) == 1, 'NaNs in dataset'
    assert len(np.unique(np.isnan(test[:,:,4]))) == 1, 'NaNs in dataset'

def voxelize(ISOTOPE, sample_size):
    """
    Begins the voxelization process.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. C16).

    sample_size : int
        The number of instances you want each event to become.
    
    Returns
    ----------
    None.
    """
        
    name = ISOTOPE + '_size' + str(sample_size)
    data = np.load('../voxel_data/' + name + '.npy')
    
    # Define the ranges for normalization
    RANGES = {
        'MIN_X': -270.0,
        'MAX_X': 270.0,
        'MIN_Y': -270.0,
        'MAX_Y': 270.0,
        'MIN_Z': -185.0,
        'MAX_Z': 1155.0,  # Corrected from previous example
        'MIN_LOG_A': 0.0,
        'MAX_LOG_A': 8.60
    }
    
    # Print maximums and minimums from the original dataset
    print("Maximums and minimums from the original data set:")
    print(np.amax(data[:,:,0]), np.amax(data[:,:,1]), np.amax(data[:,:,2]))
    print(np.amin(data[:,:,0]), np.amin(data[:,:,1]), np.amin(data[:,:,2]))
    
    # Normalize the x, y, z values correctly
    data[:,:,0] = (data[:,:,0] - RANGES['MIN_X']) / (RANGES['MAX_X'] - RANGES['MIN_X'])
    data[:,:,1] = (data[:,:,1] - RANGES['MIN_Y']) / (RANGES['MAX_Y'] - RANGES['MIN_Y'])
    data[:,:,2] = (data[:,:,2] - RANGES['MIN_Z']) / (RANGES['MAX_Z'] - RANGES['MIN_Z'])
    
    # Print normalized maximums and minimums
    print("\nNormalized maximums and minimums:")
    print(np.amax(data[:,:,0]), np.amax(data[:,:,1]), np.amax(data[:,:,2]))
    print(np.amin(data[:,:,0]), np.amin(data[:,:,1]), np.amin(data[:,:,2]))
    
    # Save the normalized data
    new_array_name = ISOTOPE + '_size' + str(sample_size) + '_sampled_normal'
    np.save('../voxel_data/' + new_array_name, data)

def label(ISOTOPE, sample_size, K_x, K_y, K_z):
    """
    Saves the data along with the voxels each instance belongs to.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. C16).

    sample_size : int
        The number of instances you want each event to become.

    K_x, K_y, K_z : ints
        Number of splits along each axis.
    
    Returns
    ----------
    None.
    """
    
    name = ISOTOPE + '_size' + str(sample_size) + '_sampled_normal' # Using normalized data
    data = np.load('../voxel_data/' + name + '.npy')
    
    K_x = 2
    K_y = 2
    K_z = 6
    
    new_data = np.zeros((len(data), sample_size, 6), float)
    
    #Store voxel bounds in dict with keys being lists
    #Structure: voxel[key] = [ [voxel_lower_bounds], [voxel_upper_bounds], voxel_id]
    voxels = dict({})
    i=0
    
    for z in range(K_z):
        for y in range(K_y):
            for x in range(K_x):
                key = [x,y,z]
                value = []
                #Creating lower bound of voxel
                min_bounds = [-1,-1,-1]
                min_bounds[0] = (1/K_x)*x
                min_bounds[1] = (1/K_y)*y
                min_bounds[2] = (1/K_z)*z
                #Creating upper bound of voxel
                max_bounds = [-2,-2,-2]
                max_bounds[0] = (1/K_x) + (1/K_x)*x
                max_bounds[1] = (1/K_y) + (1/K_y)*y
                max_bounds[2] = (1/K_z) + (1/K_z)*z
                
                value.append(min_bounds)
                value.append(max_bounds)
                value.append(i)
                i += 1
                
                voxels[str(key)] = value
    
    # Identifying voxel id for each point and normalizing all points to be within [1/K_x,1/K_y,1/K_z]
    
    for i in tqdm.tqdm(range(len(data))):
        for j in range(sample_size):
    
            #Finding Current Point's Voxel Key
            voxel_key = [-1,-1,-1]
            x_val = data[i,j,0]
            if x_val < (1/K_x):
                voxel_key[0] = 0
            else:
                voxel_key[0] = 1
    
            y_val = data[i,j,1]
            if y_val < (1/K_y):
                voxel_key[1] = 0
            else:
                voxel_key[1] = 1
    
            z_val = data[i,j,2]
            voxel_key[2] = int(np.floor(z_val*K_z))
    
            #Getting related voxel info
            lower_bound = voxels[str(voxel_key)][0]
            upper_bound = voxels[str(voxel_key)][1]
            voxel_num = voxels[str(voxel_key)][2]
    
            #Normalizing coords
            new_x = x_val-lower_bound[0]
            new_y = y_val-lower_bound[1]
            new_z = z_val-lower_bound[2]
    
            #Saving new voxel coords and id
            new_data[i,j,0] = new_x
            new_data[i,j,1] = new_y
            new_data[i,j,2] = new_z
            new_data[i,j,3] = voxel_num
            new_data[i,j,4] = data[i,j,5]
            new_data[i,j,5] = str(i)
    
            data[i,j,3] = voxel_num
            data[i,j,4] = data[i,j,5]
            data[i,j,5] = str(i)
            
            
            # before: x, y, z, amplitude - 3, event index - 4, event length - 5
    
            # after: x, y, z, voxel id - 3, amplitude - 4, event length - 5
    
    print(np.amax(new_data[:,:,0]), np.amax(new_data[:,:,1]), np.amax(new_data[:,:,2]), np.amax(new_data[:,:,3]), np.amax(new_data[:,:,4]))
    print(np.amin(new_data[:,:,0]), np.amin(new_data[:,:,1]), np.amin(new_data[:,:,2]), np.amin(new_data[:,:,3]), np.amin(new_data[:,:,4]))
      
    # Converting all voxel keys from list to corresponding voxel id
    i = 0
    
    for z in range(K_z):
        for y in range(K_y):
            for x in range(K_x):
                key = [x,y,z]
                voxels[i] = voxels[str(key)]
                del voxels[str(key)]
                i += 1
                
    new_array_name1 = ISOTOPE + '_size' + str(sample_size) + '_base_voxels.npy'
    new_array_name2 = ISOTOPE + '_size' + str(sample_size) + '_voxelated.npy'
    
    np.save('../voxel_data/' + new_array_name1, new_data)
    np.save('../voxel_data/' + new_array_name2, data[:,:,:6]) # THIS is the file to use for incorporating unshuffled data in training set.

    voxels_np = np.zeros((K_x * K_y * K_z,2,3))
    for i in range(K_x * K_y * K_z):
        min_bounds = voxels[i][0]
        max_bounds = voxels[i][1]
        voxels_np[i,0] = min_bounds
        voxels_np[i,1] = max_bounds
    
    # print(voxels)
    # print(voxels_np)
    np.save('../voxel_data/voxel_bounds.npy', voxels_np)
    
    name1 = ISOTOPE + '_size' + str(512) + '_voxelated'
    name2 = ISOTOPE + '_size' + str(512) + '_base_voxels'
    voxel_data = np.load('../voxel_data/' + name1 + '.npy')
    next_step_data = np.load('../voxel_data/' + name2 + '.npy')
    
    print(voxel_data.shape, next_step_data.shape)

    return voxels
    

def shuffle(ISOTOPE, sample_size, voxels, K_x, K_y, K_z):
    """
    Responsible for shuffling the voxels of each individual event.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. C16).

    sample_size : int
        The number of instances you want each event to become.

    K_x, K_y, K_z : ints
        Number of splits along each axis.
    
    Returns
    ----------
    None.
    """
    
    name = ISOTOPE + '_size' + str(sample_size) + '_base_voxels'
    name_unshuffled = ISOTOPE + '_size' + str(sample_size) + '_voxelated'
    data = np.load('../voxel_data/' + name + '.npy')           # This is the data file with all of the voxels plotted on each other in one voxel. This makes the shuffling process easier
    data_unshuffled = np.load('../voxel_data/' + name_unshuffled + '.npy')           # This is the original data with the normalized, voxelized event

    new_data = np.zeros((len(data), sample_size, 6), float)
    for i in tqdm.tqdm(range(len(data))):
        
        #Gets a list of where each voxel is going to be shuffled
        #Repeats until each voxel is assigned an id other than its own
        flag = True
        while flag:
            permutations = []
            ids = []
            overlap = False
            for j in range(K_x * K_y * K_z):
                ids.append(j)
            for j in range(K_x * K_y * K_z):
                val = random.choice(ids)
                permutations.append(val)
                ids.remove(val)
            for j,x in enumerate(permutations):
                if j == x:
                    overlap = True
            if overlap == False:
                flag = False
                
        #Moves each point to its new voxel
        for j in range(sample_size):
            augment = 0
            old_id = data[i,j,3]
            length_of_event = data[i,j,4]
            event_num = data[i,j,5]
            
            new_id = permutations[int(old_id)]
            new_min_bounds = voxels[new_id][0]
            new_x = data[i,j,0] + new_min_bounds[0] + augment
            new_y = data[i,j,1] + new_min_bounds[1] + augment
            new_z = data[i,j,2] + new_min_bounds[2] + augment
            
            #Doesn't include augment if causes point to move out of unit cube
            if (new_x < 0) or (new_x > 1):
                new_x = data[i,j,0] + new_min_bounds[0]
            if (new_y < 0) or (new_y > 1):
                new_y = data[i,j,1] + new_min_bounds[1]
            if (new_z < 0) or (new_z > 1):
                new_z = data[i,j,2] + new_min_bounds[2]
    
            new_data[i,j,0] = new_x
            new_data[i,j,1] = new_y
            new_data[i,j,2] = new_z
            new_data[i,j,3] = old_id
            new_data[i,j,4] = length_of_event
            new_data[i,j,5] = event_num
    
    final_data = np.concatenate((data_unshuffled[:1000],new_data),axis=0) 
    # Use the above line to change the proportions of unshuffled and shuffled data in the training set
    
    print(final_data.shape)
    # to confirm concatenation occured correctly
    
    shuffled_data = ISOTOPE + '_size' + str(sample_size) + '_shuffled_voxels_only' # Only shuffled
    np.save('../voxel_data/' + shuffled_data, new_data)
    
    new_array_name = ISOTOPE + '_size' + str(sample_size) + '_shuffled_voxels' # Includes unshuffled
    np.save('../voxel_data/' + new_array_name, final_data)

def test_train_and_val(ISOTOPE, sample_size):
    """
    Performs a 20-test 20-val 60-train split on all 4-track events.
    
    Parameters
    ----------
    ISOTOPE : str
        Name of the isotope (ex. C16).

    sample_size : int
        The number of instances you want each event to become.
    
    Returns
    ----------
    None.
    """
    
    # generates an array of numbers as long as the length of the data to randomize the events 
    name = ISOTOPE + '_size' + str(sample_size)
    all_events = np.load('../voxel_data/' + name + '_shuffled_voxels.npy')
    rand_shuffle = np.random.choice(len(all_events), len(all_events), replace = False)
    
    
    # 20-20 marking for test and validation
    test_split = int(len(all_events) * .2)
    val_split = int(len(all_events) * .4)
    
    
    test_data =  all_events[rand_shuffle[:test_split],:,:]    #only saving the indices and number of tracks of the test events
    val_data = all_events[rand_shuffle[test_split:val_split],:,:]
    train_data = all_events[rand_shuffle[val_split:],:,:]
    
    
    print(test_data.shape, val_data.shape, train_data.shape)
    np.save('../voxel_data/' + ISOTOPE + '_size' + str(sample_size) + 'test', test_data)
    np.save('../voxel_data/' + ISOTOPE + '_size' + str(sample_size) + 'train', train_data)
    np.save('../voxel_data/' + ISOTOPE + '_size' + str(sample_size) + 'val', val_data)

def main():
    
    # user inputs
    sample_size = 512 # enter the size to which events will be up/downsampled
    TRACK_CLASS = False
    dimension = 4 # desired dimension of data to be input
    ISOTOPE = 'C16+O16'
    min_points_threshold = 60 # Determine threshold for minimum number of non-zero points (to be used in the filter_data function)
    min_charge_threshold = 0 # ^ same but for charge value

    K_x = 2
    K_y = 2
    K_z = 6

    # calling the functions
    C16_data, C16_event_lens, O16_data, O16_event_lens = load_data()
    C16_valid_event_indices, C16_filtered_data = filter_point_clouds_C16(C16_data)
    O16_valid_event_indices, O16_filtered_data = filter_point_clouds_O16(O16_data)
    combine_data(C16_filtered_data, C16_valid_event_indices, C16_event_lens, O16_filtered_data, O16_valid_event_indices, O16_event_lens)
    random_sample(ISOTOPE, sample_size, dimension)
    scale_and_split(ISOTOPE, sample_size)
    voxelize(ISOTOPE, sample_size)
    voxels = label(ISOTOPE, sample_size, K_x, K_y, K_z)
    shuffle(ISOTOPE, sample_size, voxels, K_x, K_y, K_z)
    test_train_and_val(ISOTOPE, sample_size)
    

if __name__ == "__main__":
    main()