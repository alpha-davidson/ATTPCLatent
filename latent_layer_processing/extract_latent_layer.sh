#!/bin/bash
#SBATCH --job-name "LAT_LAY_EXTR"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
PYTHONPATH=../.. python3 -m ATTPCLatent.latent_layer_processing.global_features_extraction.py --beam O16 --num-classes 24 ../relative/path/to/model ../relative/path/to/data