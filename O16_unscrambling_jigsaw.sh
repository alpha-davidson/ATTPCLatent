#!/bin/bash
#SBATCH --job-name "full save O16 shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python O16_pretrain_on_jigsaw_events.py --num-classes 27 --num-epochs 800 O16_expt_downstream/voxel_data/O16_size512