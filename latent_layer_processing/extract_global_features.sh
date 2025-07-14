#!/bin/bash
#SBATCH --job-name "LAT_LAY_EXTR"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_extraction ../training/O16_models/2025-06-16-14:56:34/full_model ../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512 global_features/O16_3track.npy
