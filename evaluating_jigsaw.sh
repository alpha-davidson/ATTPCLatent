#!/bin/bash
#SBATCH --job-name "Jigsaw"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2023-06-12-14:03:26/weights/cp-084.ckpt \
          voxel_data/Mg22_size512
          
