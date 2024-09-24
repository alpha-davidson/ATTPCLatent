#!/bin/bash
#SBATCH --job-name "full save mg22 shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python Mg22_pretrain_on_jigsaw_events.py --num-classes 24 --num-epochs 100 Mg22_voxel_data/Mg22_size512
