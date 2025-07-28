#!/bin/bash
#SBATCH --job-name "Mg22_Convert_The_Data"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Calling this in the terminal will run the file 'Mg22_voxel_pipeline.py'
python3 Mg22_voxel_pipeline.py