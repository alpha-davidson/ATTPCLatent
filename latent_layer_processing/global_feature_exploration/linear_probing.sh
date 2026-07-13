#!/bin/bash
#SBATCH --job-name "LINEAR_PROBING"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --output=logs/linear_probe_%j.log

source activate attpc-latent

# Arguments:
# --name: label used for the output folder name, saved under ./linear_probe_results/<name>_linear_probe
# --test-size: fraction of the dataset held out for testing, e.g. 0.2 means 20 percent test data
# --seed: optional integer seed for reproducible train/test splits, e.g. --seed 42 (default is None, and you will have to add this line if you want to use a specific seed)
# --task: classification (default) or regression
# --regularization: classification uses C (larger = weaker regularization); regression uses Ridge alpha
# --classifier: logistic-regression (default) or linear-svm (classification only)
# --class-name: optional display label for a class. Repeat once per sorted unique label. (default is None, and you will have to add this line if you want to use specific class names)
#               If omitted, labels default to numeric values like "0", "1", "2".
#               Example format:
#               --class-name "0,1,2 Tracks" --class-name "3 Tracks" --class-name "4,5 Tracks" 
#               Add those lines before the positional paths in the command below if needed.
# Positional paths:
#   1. features .npy file with shape (N, D) (default is ../../data/features.npy)
#   2. labels .npy file with shape (N,) (default is ../../data/master_labels.npy)
python linear_probing.py \
    --name testing \
    --test-size 0.2 \
    --task regression \
    --regularization 1.0 \
    ../../data/features.npy \
    ../../data/master_labels.npy
