#!/bin/bash
#SBATCH --job-name "Mg22_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
# python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2024-04-17-16:35:23/weights/cp-081.ckpt \voxel_data/Mg22_size512
python3 evaluate_jigsaw_reconstruction.py --num-classes 24 models/2024-04-27-22:46:06/weights/cp-507.ckpt voxel_data/Mg22_size512
