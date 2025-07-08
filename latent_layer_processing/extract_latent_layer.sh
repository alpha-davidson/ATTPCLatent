#!/bin/bash
#SBATCH --job-name "LAT_LAY_EXTR"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_extraction ../training/O16_models/2025-06-20-09:25:46/full_model
