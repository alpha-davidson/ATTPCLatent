#!/bin/bash
#SBATCH --job-name "LINEAR_PROBING"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_exploration.linear_probing --beam O16 --num-classes 3 ../../training/O16_models/2025-06-16-14:56:34/full_model ../global_features/O16_Exp_extr.npy ../global_features/O16_Exp_labels.npy
