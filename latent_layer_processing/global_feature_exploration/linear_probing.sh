#!/bin/bash
#SBATCH --job-name "LINEAR_PROBING"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_exploration.linear_probing --beam O16 --num-classes 24 ../../training/O16_models/2025-06-16-14:56:34/full_model ../../simulated_data/process_data/O16/voxel_data/output_digi_HDF_2Body/O16_size512 ../../simulated_data/process_data/O16/voxel_data/output_digi_HDF_3Body/O16_size512 