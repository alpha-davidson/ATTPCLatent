import os
import sys
import click
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from matplotlib import colors
import matplotlib.cm as cm
from svm import svm_classify 

def plot_event(fig, sampled_data, event_num):
    # Define the colormap and normalization based on the entire dataset
    colormap = 'viridis'
    norm = plt.Normalize(vmin=0, vmax=500)

    # Create a ScalarMappable object that maps normalized values to colors
    scalarmappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    
    x = sampled_data[event_num, :, 0].flatten()
    y = sampled_data[event_num, :, 1].flatten()
    z = sampled_data[event_num, :, 2].flatten()

    # Create a 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-250, 250)
    
    # Creating a plot
    ax.scatter(x, z, y, s=5, c=sampled_data[event_num, :, 3].flatten(), cmap=colormap, norm=norm) 

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_zlabel('Y (mm)')
    ax.set_title(f'Event: {event_num}')
    
    # Fixing axis label cutoff
    ax.set_box_aspect(aspect=None, zoom=0.85)  
        
    # Colorbar
    fig.colorbar(scalarmappable, label='Energy Loss', pad=0.01, shrink=0.2, ax=ax,
                spacing='proportional')

def plot_classification_accuracy(): 
    X = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 140]
    y = []
    
    for i in X:
        y.append(svm_classify(i))

    fig = plt.figure(figsize=(7.4,7.4))
    plt.plot(X, y)

    # Set labels and title
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')

    for i, label in enumerate(X):
        label = f"{round((label*6*100)/2458, 1)}%"
        plt.annotate(label, (X[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig(f'plots/sample_accuracy.png')

def plot_data():
    # sampled_data = np.load('../../data_processing/O16/voxel_data/O16_size512.npy')

    # create a folder for plots
    folder_path = f'plots/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_classification_accuracy()
    # track2_nums = [2, 4, 6, 7, 714]
    # folder_path = f'plots/track2'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # for i in track2_nums:
    #     fig = plt.figure(figsize=(10, 10))
    #     plot_event(fig, sampled_data, i)
    #     plt.savefig(f'plots/track2/event_{i}.png')


    

if __name__ == '__main__':
    plot_data()
