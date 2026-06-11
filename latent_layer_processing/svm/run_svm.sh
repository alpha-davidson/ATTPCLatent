#!/bin/bash
#SBATCH --job-name "SVM_Training"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. 
source $(conda info --base)/etc/profile.d/conda.sh
conda activate attpc-latent

FEATURES_FILE="../../data/features.npy"
LABELS_FILE="../../data/master_labels.npy"
OUTPUT_DIR="./results/O16_svm_eval"
SAMPLE_COUNT=250

python3 svm.py "$FEATURES_FILE" "$LABELS_FILE" \
    --samples $SAMPLE_COUNT \
    --output-dir "$OUTPUT_DIR"
