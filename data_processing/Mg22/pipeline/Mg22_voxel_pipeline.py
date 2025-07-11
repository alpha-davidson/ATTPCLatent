import random
import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
import math
import sys
import os

def convert_data(file, original_keys, original_length):
    """
    Extracts event lengths and converts event data into a numpy array with 13 features.
    
    Features: x, y, z, time, amplitude, trackID, pointID, energy, energy_loss, angle, mass, atomic_number, event_id
    
    Parameters:
        file (h5py.File): HDF5 file containing event data.
        original_keys (list): List of event keys in the HDF5 file.
        original_length (int): Number of events.
    
    Returns:
        np.ndarray: Array of event lengths.
    """
    event_lens = np.zeros(original_length, dtype=int)
    for i, key in enumerate(original_keys):
        event_lens[i] = len(file[key])
    np.save('../voxel_data/Mg22_event_lens.npy', event_lens)
    
    max_event_length = np.max(event_lens)
    event_data = np.zeros((original_length, max_event_length, 13), dtype=float)
    
    for n in tqdm.tqdm(range(original_length), desc="Converting events"):
        event = file[original_keys[n]]
        for i, point in enumerate(event):
            event_data[n, i, :12] = list(point)
            event_data[n, i, -1] = float(n)
    
    np.save('../voxel_data/Mg22_w_key_index.npy', event_data)
    return event_lens

def random_sample(ISOTOPE, sample_size, original_length, min_points_threshold):
    """
    Samples a fixed number of points from each event, prioritizing shortest tracks.
    
    Features: x, y, z, time, amplitude, trackID, pointID, energy, energy_loss, angle, mass, atomic_number, event_id, num_tracks
    
    Parameters:
        ISOTOPE (str): Isotope name (e.g., 'Mg22').
        sample_size (int): Number of points to sample per event.
        original_length (int): Number of events.
        min_points_threshold (int): Minimum number of points required for an event.
    """
    data_array = f'{ISOTOPE}_w_key_index.npy'
    new_array_name = f'{ISOTOPE}_size{sample_size}_sampled'
    data = np.load(f'../voxel_data/{data_array}')
    event_lens = np.load('../voxel_data/Mg22_event_lens.npy')
    new_data = np.zeros((original_length, sample_size, 14), dtype=float)
    
    for i in tqdm.tqdm(range(original_length), desc="Sampling events"):
        ev_len = event_lens[i]
        if ev_len < min_points_threshold:
            continue
            
        particle_ids = data[i, :ev_len, 5]
        label, distr = np.unique(particle_ids, return_counts=True)
        shortest = label[np.argmin(distr)]
        shortest_ind = np.where(particle_ids == shortest)[0]
        
        if ev_len == sample_size:
            new_data[i, :, :-1] = data[i, :ev_len, :]
        else:
            instant = 0
            for n in shortest_ind:
                new_data[i, instant, :-1] = data[i, n, :]
                instant += 1
            need = sample_size - len(shortest_ind)
            random_points = np.random.choice(range(ev_len), need, replace=need > ev_len)
            for r in random_points:
                new_data[i, instant, :-1] = data[i, r, :]
                instant += 1
        
        unique_point_ids = np.unique(data[i, :ev_len, 5])
        new_data[i, 0, -1] = len(unique_point_ids) - 1
    
    np.save(f'../voxel_data/{new_array_name}.npy', new_data)

def filter_data(ISOTOPE, sample_size, original_length):
    """
    Filters events to include only those with exactly 4 tracks.
    
    Parameters:
        ISOTOPE (str): Isotope name.
        sample_size (int): Number of points per event.
        original_length (int): Original number of events.
    """
    file_name = f'{ISOTOPE}_size{sample_size}_sampled'
    new_file_name = f'{ISOTOPE}_size{sample_size}_even_tracks_sampled.npy'
    raw_data = np.load(f'../voxel_data/{file_name}.npy')
    new_data = np.zeros((original_length, sample_size, 14), dtype=float)
    count = 0
    
    for i in tqdm.tqdm(range(original_length), desc="Filtering events"):
        event = raw_data[i]
        unique_point_ids = np.unique(event[:, 5])
        current_tracks = len(unique_point_ids) - 1
        og_tracks = event[0, -1]
        
        if og_tracks in [0, 4] or og_tracks != current_tracks:
            continue
        
        event[:, 5] -= 1
        new_data[count] = event
        count += 1
    
    print(f"Number of remaining events: {count}")
    np.save(f'../voxel_data/{new_file_name}', new_data[:count])

def voxelize(ISOTOPE, sample_size):
    """
    Normalizes spatial coordinates and assigns points to voxels.
    
    Parameters:
        ISOTOPE (str): Isotope name.
        sample_size (int): Number of points per event.
    """
    name = f'{ISOTOPE}_size{sample_size}_even_tracks_sampled'
    data = np.load(f'../voxel_data/{name}.npy')
    
    max_x, max_y, max_z = 255, 255, 1000
    min_x, min_y, min_z = -255, -255, -60
    
    print("Maximums and minimums from the original data set:")
    print(np.amax(data[:, :, 0]), np.amax(data[:, :, 1]), np.amax(data[:, :, 2]))
    print(np.amin(data[:, :, 0]), np.amin(data[:, :, 1]), np.amin(data[:, :, 2]))
    
    data[:, :, 0] = (data[:, :, 0] + abs(min_x)) / (max_x + abs(min_x))
    data[:, :, 1] = (data[:, :, 1] + abs(min_y)) / (max_y + abs(min_y))
    data[:, :, 2] = (data[:, :, 2] + abs(min_z)) / (max_z + abs(min_z))
    
    print("\nNormalized maximums and minimums:")
    print(np.amax(data[:, :, 0]), np.amax(data[:, :, 1]), np.amax(data[:, :, 2]))
    print(np.amin(data[:, :, 0]), np.amin(data[:, :, 1]), np.amin(data[:, :, 2]))
    
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_sampled_normal.npy', data)
    
    K_x, K_y, K_z = 2, 2, 6
    voxels = {}
    i = 0
    for z in range(K_z):
        for y in range(K_y):
            for x in range(K_x):
                key = f"{x},{y},{z}"
                min_bounds = [x/K_x, y/K_y, z/K_z]
                max_bounds = [(x+1)/K_x, (y+1)/K_y, (z+1)/K_z]
                voxels[key] = [min_bounds, max_bounds, i]
                i += 1
    
    new_data = np.zeros((len(data), sample_size, 6), dtype=float)
    
    for i in tqdm.tqdm(range(len(data)), desc="Voxelizing events"):
        for j in range(sample_size):
            x_val, y_val, z_val = data[i, j, :3]
            voxel_key = [
                0 if x_val < 1/K_x else 1,
                0 if y_val < 1/K_y else 1,
                int(np.floor(z_val * K_z))
            ]
            voxel_info = voxels[f"{voxel_key[0]},{voxel_key[1]},{voxel_key[2]}"]
            
            new_x = x_val - voxel_info[0][0]
            new_y = y_val - voxel_info[0][1]
            new_z = z_val - voxel_info[0][2]
            
            new_data[i, j, 0] = new_x
            new_data[i, j, 1] = new_y
            new_data[i, j, 2] = new_z
            new_data[i, j, 3] = voxel_info[2]
            new_data[i, j, 4] = data[i, j, -1]
            new_data[i, j, 5] = i
            
            data[i, j, 3] = voxel_info[2]
            data[i, j, 4] = data[i, j, -1]
            data[i, j, 5] = i
    
    print("\nVoxelized coordinate ranges:")
    print(np.amax(new_data[:, :, 0]), np.amax(new_data[:, :, 1]), np.amax(new_data[:, :, 2]))
    print(np.amin(new_data[:, :, 0]), np.amin(new_data[:, :, 1]), np.amin(new_data[:, :, 2]))
    
    i = 0
    for z in range(K_z):
        for y in range(K_y):
            for x in range(K_x):
                key = f"{x},{y},{z}"
                voxels[i] = voxels[key]
                del voxels[key]
                i += 1
    
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_base_voxels.npy', new_data)
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_voxelated.npy', data[:, :, :6])

def shuffle(ISOTOPE, sample_size):
    """
    Shuffles points between voxels while preserving event structure.
    
    Parameters:
        ISOTOPE (str): Isotope name.
        sample_size (int): Number of points per event.
    """
    name = f'{ISOTOPE}_size{sample_size}_base_voxels'
    name_unshuffled = f'{ISOTOPE}_size{sample_size}_voxelated'
    data = np.load(f'../voxel_data/{name}.npy')
    data_unshuffled = np.load(f'../voxel_data/{name_unshuffled}.npy')
    new_data = np.zeros((len(data), sample_size, 6), dtype=float)
    
    K_x, K_y, K_z = 2, 2, 6
    voxels = {}
    voxel_id = 0
    for z in range(K_z):
        for y in range(K_y):
            for x in range(K_x):
                voxels[voxel_id] = [x/K_x, y/K_y, z/K_z]
                voxel_id += 1
    
    for i in tqdm.tqdm(range(len(data)), desc="Shuffling voxels"):
        flag = True
        while flag:
            permutations = []
            ids = list(range(K_x * K_y * K_z))
            for _ in range(K_x * K_y * K_z):
                val = random.choice(ids)
                permutations.append(val)
                ids.remove(val)
            flag = any(j == x for j, x in enumerate(permutations))
        
        for j in range(sample_size):
            old_id = int(data[i, j, 3])
            num_tracks = data[i, j, 4]
            event_num = data[i, j, 5]
            new_id = permutations[old_id]
            new_min_bounds = voxels[new_id]
            
            new_x = data[i, j, 0] + new_min_bounds[0]
            new_y = data[i, j, 1] + new_min_bounds[1]
            new_z = data[i, j, 2] + new_min_bounds[2]
            
            if not (0 <= new_x <= 1):
                new_x = data[i, j, 0] + new_min_bounds[0]
            if not (0 <= new_y <= 1):
                new_y = data[i, j, 1] + new_min_bounds[1]
            if not (0 <= new_z <= 1):
                new_z = data[i, j, 2] + new_min_bounds[2]
                
            new_data[i, j, 0] = new_x
            new_data[i, j, 1] = new_y
            new_data[i, j, 2] = new_z
            new_data[i, j, 3] = old_id
            new_data[i, j, 4] = num_tracks
            new_data[i, j, 5] = event_num
    
    final_data = np.concatenate((data_unshuffled[:1000], new_data), axis=0)
    print(f"Final dataset shape: {final_data.shape}")
    
    shuffled_data = f'{ISOTOPE}_size{sample_size}_shuffled_voxels_only'
    np.save(f'../voxel_data/{shuffled_data}.npy', new_data)
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_shuffled_voxels.npy', final_data)

def train_val_test(ISOTOPE, sample_size):
    """
    Splits data into train (60%), validation (20%), and test (20%) sets.
    
    Parameters:
        ISOTOPE (str): Isotope name.
        sample_size (int): Number of points per event.
    """
    name = f'{ISOTOPE}_size{sample_size}'
    all_events = np.load(f'../voxel_data/{name}_shuffled_voxels.npy')
    rand_shuffle = np.random.choice(len(all_events), len(all_events), replace=False)
    
    test_split = int(len(all_events) * 0.2)
    val_split = int(len(all_events) * 0.4)
    
    test_data = all_events[rand_shuffle[:test_split]]
    val_data = all_events[rand_shuffle[test_split:val_split]]
    train_data = all_events[rand_shuffle[val_split:]]
    
    print(f"Test: {test_data.shape}, Validation: {val_data.shape}, Train: {train_data.shape}")
    
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_test.npy', test_data)
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_train.npy', train_data)
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_val.npy', val_data)

    folder_path = f'../data_splits'
    os.makedirs(folder_path, exist_ok=True)
    
    np.save(f'../data_splits/{ISOTOPE}_size{sample_size}_test.npy', test_data)
    np.save(f'../data_splits/{ISOTOPE}_size{sample_size}_train.npy', train_data)
    np.save(f'../data_splits/{ISOTOPE}_size{sample_size}_val.npy', val_data)
    

def rebalance(ISOTOPE, sample_size):
    """
    Duplicates events with underrepresented voxels to balance training data.
    
    Parameters:
        ISOTOPE (str): Isotope name.
        sample_size (int): Number of points per event.
    """
    tr_data = np.load(f'../voxel_data/{ISOTOPE}_size{sample_size}_train.npy')
    appended_data = []
    
    vids = tr_data[:, :, 3]
    invalid_values = np.any(~np.isin(vids, [4.0, 13.0, 22.0]), axis=1)
    
    for i in np.where(invalid_values)[0]:
        appended_data.append(tr_data[i])
    
    if appended_data:
        for _ in range(3):
            tr_data = np.concatenate((tr_data, np.array(appended_data)), axis=0)
        np.random.shuffle(tr_data)
    
    np.save(f'../voxel_data/{ISOTOPE}_size{sample_size}_train_dup.npy', tr_data)

def main():
    """
    Main function to process Mg22 data through all stages.
    """
    sample_size = 512
    CLASSIFICATION = 'Voxel'
    PROJECTION = 'XYZ'
    ISOTOPE = 'Mg22'
    min_points_threshold = 40
    
    data = h5py.File('../voxel_data/output_digi_HDF_Mg22_Ne20pp_8MeV.h5', 'r')
    original_keys = list(data.keys())
    original_length = len(original_keys)
    
    event_lens = convert_data(data, original_keys, original_length)
    random_sample(ISOTOPE, sample_size, original_length, min_points_threshold)
    filter_data(ISOTOPE, sample_size, original_length)
    voxelize(ISOTOPE, sample_size)
    shuffle(ISOTOPE, sample_size)
    train_val_test(ISOTOPE, sample_size)
    rebalance(ISOTOPE, sample_size)
    
    data.close()

if __name__ == "__main__":
    main()