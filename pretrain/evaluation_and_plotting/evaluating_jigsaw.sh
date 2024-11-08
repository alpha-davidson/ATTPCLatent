#!/bin/bash
#SBATCH --job-name "JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
python3 evaluate_jigsaw_reconstruction.py --beam C16+O16 --num-classes 24 ../training/C16+O16_models/2024-11-05-13:30:48/weights/cp-073.ckpt ../data_processing/C16+O16/voxel_data/C16+O16_size512
