import h5py as h5
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

DATA_PREFIX = "C16_size256_"

#change sample_size to match ground-truth data size, will depend on what new C16-data looks like. Or new isotope data.

def merge_data(sample_size=768):
    rng = np.random.default_rng()
    
    VOXEL_PATH = "../data/"
    CLUSTER_PATH = f"{VOXEL_PATH}example_run_0017_clusters.h5"
    DATAFRAME_PATH = f"{VOXEL_PATH}example_run_0017_fit_2H.parquet"
    cluster_file = h5.File(CLUSTER_PATH, 'r')
    dataframe = pd.read_parquet(DATAFRAME_PATH)
    
    root_group: h5.Group = cluster_file['cluster']
    min_event = root_group.attrs.get("min_event")
    max_event = root_group.attrs.get("max_event")
    event_nums = range(min_event, max_event+1)
    
    labels_df = dataframe[['polar', 'azimuthal', 'brho', 'event', 'cluster_index']]

    features = []
    labels = []
    events_included = []
    
    for event_num in tqdm(event_nums):
        event = root_group[f"event_{event_num}"]
        n_clusters = event.attrs.get('nclusters')
        
        event_data = [] # temp placeholder to combine tracks before sampling
        
        # Combine the tracks into a single point cloud
        for i in range(n_clusters):
            track = event[f"cluster_{i}"]["cloud"]
            track_labels = labels_df.loc[(labels_df["event"] == event_num) & (labels_df["cluster_index"] == i)]
            
            if not track_labels.empty: 
                
                # We have labels for this track, so include them
                num_points = track.shape[0]
                track_labels = np.tile(np.array(track_labels[['polar', 'azimuthal', 'brho']]).flatten(), (num_points, 1))
                track_data = np.concatenate((track, track_labels), axis=1)
                event_data.append(track_data)

            
            
        # If we have any data points for this event number
        if (event_data != []):
            event_data = np.concatenate(event_data).reshape(-1, 7)
            sampled_point_cloud = rng.choice(np.array(event_data), sample_size)
            features.append(sampled_point_cloud[:,:4])
            labels.append(sampled_point_cloud[:,4:])
            events_included.append([event_num, n_clusters])

    if not os.path.exists(f"{VOXEL_PATH}"):
        os.mkdir(f"{VOXEL_PATH}")

    np.save(f"{VOXEL_PATH}{DATA_PREFIX}events_included.npy", np.array(events_included))
    np.save(f"{VOXEL_PATH}{DATA_PREFIX}final_f.npy", np.array(features))
    np.save(f"{VOXEL_PATH}{DATA_PREFIX}final_l.npy", np.array(labels))


merge_data()