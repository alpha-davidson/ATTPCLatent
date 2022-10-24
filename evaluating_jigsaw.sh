#!/bin/bash
#SBATCH --job-name "Jigsaw"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct time stamp
python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2022-10-21-13:12:50/weights \
          voxel_data/Mg22_size512
          