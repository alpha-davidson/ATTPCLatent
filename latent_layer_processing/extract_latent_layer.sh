#!/bin/bash
#SBATCH --job-name "LAT_LAY_EXTR"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
python3 global_features_extraction.py --beam O16 --num-classes 24 ../training/O16_models/2025-06-16-14:16:04/full_model ../data_processing/O16/voxel_data/O16_size512