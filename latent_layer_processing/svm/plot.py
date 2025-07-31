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
from svm import svm_classify, svm_neural_network_classify

def plot_event(fig, sampled_data, event_num):
    # define the colormap and normalization based on the entire dataset
    colormap = 'viridis'
    norm = plt.Normalize(vmin=0, vmax=500)

    # create a ScalarMappable object that maps normalized values to colors
    scalarmappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    
    x = sampled_data[event_num, :, 0].flatten()
    y = sampled_data[event_num, :, 1].flatten()
    z = sampled_data[event_num, :, 2].flatten()

    # create a 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-250, 250)
    
    # creating a plot
    ax.scatter(x, z, y, s=5, c=sampled_data[event_num, :, 3].flatten(), cmap=colormap, norm=norm) 

    # set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_zlabel('Y (mm)')
    ax.set_title(f'Event: {event_num}')
    
    # fixing axis label cutoff
    ax.set_box_aspect(aspect=None, zoom=0.85)  
        
    # colorbar
    fig.colorbar(scalarmappable, label='Amplitude', pad=0.01, shrink=0.2, ax=ax,
                spacing='proportional')

def plot_classification_accuracy(): 
    X = [1, 10, 50, 100, 200, 300]   # choose the number of samples for SVM 
    y = []

    # SVM f1 scores
    for i in X:
        y.append(svm_classify(i))

    fig = plt.figure(figsize=(7.4,7.4))
    plt.plot(X, y)

    # set labels and title
    plt.xlabel('Samples')
    plt.ylabel('F1')

    for i, label in enumerate(X):
        label = f"{round((label*100)/476, 1)}%"
        plt.annotate(label, (X[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig(f'plots/sample_f1.png')

def plot_data():
    # create a folder for plots
    folder_path = f'plots/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # plot f1 score in relation to the number of samples
    plot_classification_accuracy()

if __name__ == '__main__':
    plot_data()
