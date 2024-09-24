#!/bin/bash
#SBATCH --job-name "full save C16 shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 C16_pretrain_on_jigsaw_events.py --num-classes 24 --num-epochs 100 C16_pretrain/data/C16_size512
