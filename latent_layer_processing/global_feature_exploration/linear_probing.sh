#!/bin/bash
#SBATCH --job-name "LINEAR_PROBING"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --output=logs/linear_probe_%j.log

source activate attpc-latent
# Adapt the models sub-folder as needed to the correct file name. 
python linear_probing.py \
    --name testing \
    --test-size 0.2 \
    --seed 42 \
    --regularization 1.0 \
    --min-train-size 50 \
    --max-train-size 16000 \
    --num-size-points 20 \
    --cv-folds 3 \
    ../../data/features.npy \
    ../../data/master_labels.npy