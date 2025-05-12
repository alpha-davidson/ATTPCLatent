#!/bin/bash
#SBATCH --job-name "JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
python3 evaluate_jigsaw_reconstruction.py --beam O16 --num-classes 24 ../training/O16_models/2025-04-16-13:05:03/weights/cp-097.ckpt ../data_processing/O16/voxel_data/O16_size512
