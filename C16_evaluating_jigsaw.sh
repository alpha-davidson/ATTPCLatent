#!/bin/bash
#SBATCH --job-name "C16_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
python3 C16_evaluate_jigsaw_reconstruction.py --num-classes 24 C16_models/2024-10-01-11:53:16/weights/cp-039.ckpt C16_pretrain/voxel_data/C16_size512
