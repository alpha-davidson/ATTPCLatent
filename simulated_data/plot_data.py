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

@click.command()
@click.option('--beam', default='O16', type=click.STRING, help='The beam to train on (e.g. O16, Mg22)')
@click.argument('file')

def plot_data(beam, file):
    ISOTOPE = beam
    sample_size = '512'
    name = ISOTOPE + '_size' + str(sample_size)

    sampled_data = np.load(f'process_data/{ISOTOPE}/voxel_data/{file}/{ISOTOPE}_size512_sampled.npy')
    
    event_num = random.randint(1, len(sampled_data))

    for i in range (15):
        # create a folder for plots
        folder_path = f'plots/{ISOTOPE}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # create a folder for an isotope
        folder_path = f'plots/{ISOTOPE}/{name}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # create a folder for a particular file
        folder_path = f'plots/{ISOTOPE}/{name}/{file}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path) 
        
        fig = plt.figure(figsize=(10, 10))
        plot_event(fig, sampled_data, event_num)
        
        # Saving the plot
        plt.savefig(f'plots/{ISOTOPE}/{name}/{file}/event_{event_num}.png')
        event_num = random.randint(1, len(sampled_data))

if __name__ == '__main__':
    plot_data()
