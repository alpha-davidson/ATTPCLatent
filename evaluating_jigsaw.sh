#!/bin/bash
#SBATCH --job-name "O16_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
# python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2024-04-17-16:35:23/weights/cp-081.ckpt \voxel_data/Mg22_size512
python3 evaluate_jigsaw_reconstruction.py --num-classes 24 models/2024-04-17-19:52:36/weights/cp-090.ckpt \voxel_data/Mg22_size512