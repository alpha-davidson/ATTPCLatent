#!/bin/bash
#SBATCH --job-name "full save mg22 shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 pretrain_on_jigsaw_events.py --num-classes 27 --num-epochs 5 TPCNet/voxel_data/Mg22_size512
