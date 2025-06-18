#!/bin/bash
#SBATCH --job-name "full save shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

PYTHONPATH=../.. python3 -m ATTPCLatent.training.pretrain_on_jigsaw_events.py --beam O16 --batch-size 256 --num-classes 24 --num-epochs 100 ../relative/path/to/data