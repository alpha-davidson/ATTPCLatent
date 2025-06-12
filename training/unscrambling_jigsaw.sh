#!/bin/bash
#SBATCH --job-name "full save shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python pretrain_on_jigsaw_events.py --beam O16 --batch-size 64 --num-classes 24 --num-epochs 200 ../data_processing/O16/voxel_data/O16_size512