import numpy as np
import os

training_set = np.load('Mg22_pretrain/voxel_data/Mg22_size512_convertXYZQ_train.npy')
training_set.shape


rand_shuffle = np.random.choice(len(training_set), len(training_set), replace = False)
split= 0
train_data= training_set[rand_shuffle[split:],:,:]

# Assuming train_data is already defined and has shape (6000, 512, 7)

# Step 1: Create a boolean mask for the condition
mask = train_data[:, 0, 6] == 0 # change this number if a different number of tracks has an overwhelming number of events

# Step 2: Get the indices where the condition is true
indices_true = np.where(mask)[0]

# Step 3: Randomly sample 1500 indices from the true condition
sampled_indices = np.random.choice(indices_true, size=1500, replace=False) # you can replace size with whatever is necessary

# Step 4: Get the indices where the condition is false
indices_false = np.where(~mask)[0]

# Step 5: Combine the sampled indices with the false condition indices
final_indices = np.concatenate([sampled_indices, indices_false])

# Step 6: Create the final array by indexing train_data with final_indices
final_array = train_data[final_indices]

# Verify the number of events with 0 as the first element of the 7th column
num_zero_events = np.sum(final_array[:, 0, 6] == 0)

def selective_downsample(data, target_counts):
    # Get the unique classes and their counts
    classes, counts = np.unique(data[:, 0, 6], return_counts=True)
    
    # Initialize list to store indices for the final array
    final_indices = []
    
    for cls, count in zip(classes, counts):
        # Get indices for this class
        class_indices = np.where(data[:, 0, 6] == cls)[0]
        
        # If this class should be downsampled
        if cls in target_counts:
            # Randomly sample the target number of indices
            sampled_indices = np.random.choice(class_indices, size=target_counts[cls], replace=False)
            final_indices.extend(sampled_indices)
        else:
            # Keep all indices for this class
            final_indices.extend(class_indices)
    
    # Convert to numpy array and shuffle
    final_indices = np.array(final_indices)
    np.random.shuffle(final_indices)
    
    return data[final_indices]

# Define target counts for classes 0, 2, and 3
# Adjust these numbers as needed
target_counts = {
    0: 12,  # Downsample class 0 events
    1: 12,
    2: 12,  # Downsample class 2 events
    3: 12,  # Downsample class 3 events
    4: 11,
    5: 4
}


for i in range(1,11):
    # Perform selective downsampling
    downsampled_data = selective_downsample(train_data, target_counts)
    
    unique, counts = np.unique(downsampled_data[:, 0, 6], return_counts=True)
    
    train_f = downsampled_data[:,:,:6]
    train_l = downsampled_data[:,0,6]
    train_l = np.where(train_l == 5, 4, train_l)
    
    num_events = train_l.shape[0]
    new_label, new_distr = np.unique(train_l, return_counts=True)

    features_folder = f'Mg22_dw_data/Mg22_size512_{num_events}train_features'
    labels_folder = f'Mg22_dw_data/Mg22_size512_{num_events}train_labels'
    os.makedirs(features_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    np.save(f'{features_folder}/{i}',train_f)
    np.save(f'{labels_folder}/{i}',train_l)