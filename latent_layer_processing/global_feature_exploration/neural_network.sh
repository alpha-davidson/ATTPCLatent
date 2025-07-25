#!/bin/bash
#SBATCH --job-name "NeuNet_Training"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../../.. python3 -m ATTPCLatent.latent_layer_processing.global_feature_exploration.neural_network