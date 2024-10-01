#!/bin/bash
#SBATCH --job-name "full save shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python pretrain_on_jigsaw_events.py --beam Mg22 --num-classes 24 --num-epochs 100 Mg22_voxel_data/Mg22_size512