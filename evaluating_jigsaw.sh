#!/bin/bash
#SBATCH --job-name "O16_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
<<<<<<< HEAD
python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2024-03-22-00:15:14/weights/cp-083.ckpt \
=======

python3 evaluate_jigsaw_reconstruction.py --num-classes 27 TPCNet/models/2023-09-29-14:27:50/weights/cp-050.ckpt \
>>>>>>> be5dbb4bbdaee2e4c51f81e2dfd116bc54a4cbf3
          voxel_data/Mg22_size512

          