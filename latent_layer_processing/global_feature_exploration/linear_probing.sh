#!/bin/bash
#SBATCH --job-name "LINEAR_PROBING"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_exploration.linear_probing --beam O16 --num-classes 24 path/to/model path/to/first_class_ds path/to/second_class_ds 