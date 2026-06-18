#!/bin/bash
#SBATCH --job-name "LINEAR_PROBING"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --output=logs/linear_probe_%j.log

source activate attpc-latent
# Adapt the models sub-folder as needed to the correct file name. 
# IF YOU WOULD LIKE TO RUN ON A SPECIFIC SEED; ADD --seed <SEED_VALUE> TO THE ARGUMENTS BELOW. AFTER TEST-SIZE and BEFORE REGULARIZATION.
python linear_probing.py \
    --name testing \
    --test-size 0.2 \
    --regularization 1.0 \
    ../../data/features.npy \
    ../../data/master_labels.npy
