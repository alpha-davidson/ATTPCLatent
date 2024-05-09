#!/bin/bash
#SBATCH --job-name "full save O16 shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 O16_pretrain_on_jigsaw_events.py --num-classes 24 --num-epochs 50 O16_expt_downstream/voxel_data/O16_size512