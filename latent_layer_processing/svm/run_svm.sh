#!/bin/bash
#SBATCH --job-name "SVM_Training"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../../.. python3 -m ATTPCLatent.latent_layer_processing.svm.svm
