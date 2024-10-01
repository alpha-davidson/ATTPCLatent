#!/bin/bash
#SBATCH --job-name "full save shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python pretrain_on_jigsaw_events.py --beam O16 --num-classes 24 --num-epochs 100 O16_pretrain/voxel_data/O16_size512