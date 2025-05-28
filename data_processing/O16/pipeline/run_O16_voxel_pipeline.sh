#!/bin/bash
#SBATCH --job-name "O16_Convert_The_Data"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Calling this in the terminal will run the file 'O16_convert_add_data.py'
python3 O16_voxel_pipeline.py