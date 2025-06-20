#!/bin/bash
#SBATCH --job-name "Plot_Exp_Data"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Calling this in the terminal will run the file 'plot_data.py.' Also, update the name of the file from which events will be plotted.
python3 plot_data.py name_of_the_file