"""

name: O16_voxel_pipeline.py

description: Takes the raw point cloud data file called 'O16_run160.h5' and creates new files to be placed in 'O16_expt_downstream/voxel_data'. Before running, create a 'voxel_data' folder under the 'O16_expt_downstream' directory and put the 'O16_run160.h5' file into that folder.

designed for data in the following format:
x[0] ,y[1] ,z[2] ,time[3], Amplitude[4]

date created: Jul 22, 2024
date edited: Jul 22, 2024

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

# user defined functions...

def filter_data():
    

def main():
    
    # user inputs
    sample_size = 512 # enter the size to which events will be up/downsampled
    TRACK_CLASS = False
    dimension = 4 # desired dimension of data to be input
    ISOTOPE = 'O16'

    # could try to change to other isotope data in the future
    data = h5py.File('../voxel_data/O16_run160.h5','r')

    # calling the functions


if __name__ == "__main__":
    
    main()