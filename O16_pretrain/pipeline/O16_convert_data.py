import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tqdm

"""
Before running code blocks below, create a voxel_data folder under the O16_expt_downstream directory and put the O16_run160.h5 file (get that file from Raghu or Michelle) into that folder.

Note: Data files shouldn't be committed to the Github
"""

data = h5py.File('../voxel_data/O16_run160.h5','r')
keys = list(data.keys())

# # Loop through each event and print its column names
# for key in keys:
#     event = data[key]
    
#     # Check if the event is a dataset with a compound datatype
#     if isinstance(event, h5py.Dataset) and event.dtype.names:
#         print(f"Column names for {key}: {event.dtype.names}")
        
#making array of event lengths
event_lens = np.zeros(len(keys), int)
for i in range(len(keys)):
    event = keys[i]
    event_lens[i] = len(data[event])

np.save('../voxel_data/O16_event_lens.npy', event_lens)

#making a numpy array of data
event_data = np.zeros((len(keys), np.max(event_lens), 6), float)
for n in tqdm.tqdm(range(len(keys))):
    name = keys[n]
    event = data[name]
    # print(f"Name of the column {n}:", event)
    ev_len = len(event)
    for i,e in enumerate(event):
        instant = np.array(list(e))
        event_data[n,i,:5] = instant[:5]
        event_data[n,i,5] = float(n) #storing the event index
        
event_data[56437,0,5] = 56437 #fixing the empty event ('Event 60795,' index 56437)

np.save('../voxel_data/O16_w_event_keys.npy', event_data)