'''
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from matplotlib import colors
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
import os
import pickle

# plot learning curves of all metrics
def plot_learning_curves(history, filename1):
    plt.figure(figsize=(11, 6), dpi=100)
    plt.plot(history.history['loss'], 'o-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'o:', color='r', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Learning Curve(loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss']), 10), range(1, len(history.history['loss']) + 1, 10))
    plt.yscale('log')
    plt.savefig(filename1)
    
def plot_confusion_matrix(y_true, y_pred, classes, filename, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='black')
    fig.tight_layout()
    plt.savefig(filename)


# Plot histograms for distributions of labels        
def label_dist(model, filename):
    # load in dataset
    y = np.load('ATTPC_Regression/voxel_data/Mg22_size512_final_l.npy')
    
    # Data Range
    # First graph of all events disproportionately displays energy loss dist in first bin,
    # this range excludes that first bin to zoom in on smaller frequencies
    range1 = [0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040]

    # Graphing histogram
    y = np.reshape(y, (len(y) * 512, 1))
    fig, ax = plt.subplots(figsize =(7, 7))
    patches = ax.hist(y, bins=range1)
    ax.set_xlabel('Energy Loss')
    ax.set_ylabel('Frequency')
    plt.title(f'Distribution of Energy Loss for Mg22-He4 for {label_set} Events')
    plt.savefig(filename)

# Plotting point cloud comparisions
def point_clouds(model_num, path, dataset):
    # load in dataset
    sampled_data = np.load(f'{path}test_features.npy')
    label_true = np.load(f'{path}test_labels.npy')
    label_pred = np.load(f'predictions/{model_num}/{dataset}_predictions.npy')
    assert(label_true.shape == label_pred.shape)

    # Define the colormap and normalization based on the entire dataset
    colormap = 'viridis'
    norm = colors.Normalize(vmin=np.min(label_true), vmax=np.max(label_true))

    # Create a ScalarMappable object that maps normalized values to colors
    scalarmappable = cm.ScalarMappable(norm=norm, cmap=colormap)

    # Point cloud visualization loop
    for i in range(label_true.shape[0]):
        # Extract the x, y, z coordinates from the feature data
        x = sampled_data[i, :, 0].flatten()
        y = sampled_data[i, :, 1].flatten()
        z = sampled_data[i, :, 2].flatten()
        el_true = label_true[i, :, 0].flatten()
        el_pred = label_pred[i, :, 0].flatten()

        # Create 2 3D scatter plot
        fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection="3d"), tight_layout=True, figsize=(13, 13))

        # Overall Title
        plt.title(f'Point Cloud Comparisons of Predicted and Real Events')

        # Arrays for modularity
        datasets = [el_true, el_pred]
        titles = ['True', 'Predicted']

        for axis in range(len(axs)):
            # Creating plots
            axs[axis].scatter(x, z, y, s=1, c = datasets[axis])  # 's' is the marker size

            # Set labels and title
            axs[axis].set_xlabel('X (mm)')
            axs[axis].set_ylabel('Z (mm)')
            axs[axis].set_zlabel('Y (mm)')
            axs[axis].set_title(f'Point Cloud Visualization for {titles[axis]} Event: {i}')

            # Fixing axis label cutoff
            axs[axis].set_box_aspect(aspect=None, zoom=0.85)

            # Autoscale
            plt.autoscale()

            # Colorbar
            fig.colorbar(scalarmappable, label='Polar Angle', ax=axs[axis], pad=0.01, shrink=0.2, 
                    spacing='proportional')

        # Autoscale
        plt.autoscale()

        fig_path = f"point_cloud_comp/{model_num}"

        # Create a directory to save the plots if necessary
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)

        # Saving the plots
        plt.savefig(f'{fig_path}/ev_{i:04d}.png')

        # Show the plot
        plt.show()


# Instead of plotting true and predicted in different figures,
# the color of each point in the point cloud represents the squared
# difference between the true and predicted value.
# The color bar is scaled from 0 to the mean squared error across ALL events.
# Note: if multiple labels are passed in, it will only plot the first one
def squared_error_point_clouds(model_num, path, dataset):
    fig_path = f"point_cloud_comp/{model_num}"
    # Create a directory to save the plots if necessary
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    # Load in the dataset
    sampled_data = np.load(f'{path}test_features.npy')

    # Undo the scaling of the coordinate data
    sampled_data[:,:,:2] = sampled_data[:,:,:2]*550 -275
    sampled_data[:,:,2] = sampled_data[:,:,2]/1000

    label_true = np.load(f"{path}test_ground_truth.npy")
    label_pred = np.load(f"predictions/{model_num}/{dataset}_predictions.npy")
    assert(label_true.shape == label_pred.shape)
    label_shape = label_pred.shape

    # Define the function that maps scalars to colors
    MSE_all_events = np.mean((label_true-label_pred)**2)
    norm = colors.Normalize(0, MSE_all_events)
    smap = cm.ScalarMappable(norm=norm, cmap="viridis")

    for i in range(label_shape[0]):
        x = sampled_data[i,:,0].flatten()
        y = sampled_data[i,:,1].flatten()
        z = sampled_data[i,:,2].flatten()
        y_true = label_true[i,:, 0].flatten()
        y_pred = label_pred[i,:, 0].flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Put z coord data on the y axis and vice versa
        ax.scatter(x,z,y, s=1, c=(y_true-y_pred)**2, norm=norm)
        fig.colorbar(smap, ax=ax)
    
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_zlabel('Y (mm)')
        ax.set_title(f'Point-wise Squared Errors for event: {i}')
        ax.set_xlim(-275,275)
        ax.set_zlim(-275,275)
        ax.set_ylim(0,1000)    

        plt.savefig(f'{fig_path}/ev_{i:04d}.png')
