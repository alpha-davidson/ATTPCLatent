#!/bin/bash
#SBATCH --job-name "O16_JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name
# of the sub-folder matches up between the models and plots folder
python3 O16_evaluate_jigsaw_reconstruction.py --num-classes 24 O16_models/2024-06-25-10:23:39/weights/cp-034.ckpt O16_expt_downstream/voxel_data/O16_size512
