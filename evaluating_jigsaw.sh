#!/bin/bash
#SBATCH --job-name "O16_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder

python3 evaluate_jigsaw_reconstruction.py --num-classes 27 TPCNet/models/2023-09-29-14:27:50/weights/cp-050.ckpt \
          voxel_data/Mg22_size512

          