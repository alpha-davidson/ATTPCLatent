#!/bin/bash
#SBATCH --job-name "LAT_LAY_EXTR"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
python3 latent_layer.py --beam O16 --num-classes 24 ../training/O16_models/2025-06-11-13:33:45/weights/cp-200.ckpt ../data_processing/O16/voxel_data/O16_size512