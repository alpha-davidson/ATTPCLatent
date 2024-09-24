#!/bin/bash
#SBATCH --job-name "Mg22_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
python3 Mg22_evaluate_jigsaw_reconstruction.py --num-classes 24 Mg22_models/2024-06-12-11:29:56/weights/cp-082.ckpt Mg22_voxel_data/Mg22_size512
