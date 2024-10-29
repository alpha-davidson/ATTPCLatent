import numpy as np
import os
import click

training_set = np.load('data/Mg22_size512_convertXYZQ_train.npy')
training_set.shape


rand_shuffle = np.random.choice(len(training_set), len(training_set), replace = False)
split= 0
train_data= training_set[rand_shuffle[split:],:,:]

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


@click.command()
@click.option("--count-023", default=700, help="Target count for class 0, 2, 3")
@click.option("--count-1", default=52, help="Target count for class 1")
@click.option("--count-4", default=11, help="Target count for class 4")
@click.option("--count-5", default=4, help="Target count for class 5")
@click.option("--iterations", default=10, help="Number of iterations")
def main(count_023, count_1, count_4, count_5, iterations):
    # Define target counts for classes 0, 2, and 3
    # Adjust these numbers as needed
    target_counts = {
        0: count_023,  # Downsample class 0 events
        1: count_1,    # Downsample class 1 events
        2: count_023,  # Downsample class 2 events
        3: count_023,  # Downsample class 3 events
        4: count_4,    # Downsample class 4 events
        5: count_5     # Downsample class 5 events
    }
    
    
    for i in range(1, iterations + 1):
        # Perform selective downsampling
        downsampled_data = selective_downsample(train_data, target_counts)
        
        unique, counts = np.unique(downsampled_data[:, 0, 6], return_counts=True)
        
        train_f = downsampled_data[:,:,:6]
        train_l = downsampled_data[:,0,6]
        train_l = np.where(train_l == 5, 4, train_l)
        
        num_events = train_l.shape[0]
        new_label, new_distr = np.unique(train_l, return_counts=True)
    
        features_folder = f'data/Mg22_size512_{num_events}train_features'
        labels_folder = f'data/Mg22_size512_{num_events}train_labels'
        os.makedirs(features_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)
        
        np.save(f'{features_folder}/{i}',train_f)
        np.save(f'{labels_folder}/{i}',train_l)

if __name__ == "__main__":
    main()