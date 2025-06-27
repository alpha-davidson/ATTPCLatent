#!/bin/bash
#SBATCH --job-name "LAT_LAY_EXTR"
#SBATCH --mem 32G
#SBATCH --gpus 1

source /opt/conda/bin/activate "tpcnet"

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.linear_probing --beam O16 --num-classes 24 ../training/O16_models/2025-06-24-09:41:10/full_model ../data_processing/O16/simulated_data/voxel_data/output_digi_HDF_2Body/O16_size512 ../data_processing/O16/simulated_data/voxel_data/output_digi_HDF_3Body/O16_size512