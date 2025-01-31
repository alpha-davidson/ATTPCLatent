#!/bin/bash
#SBATCH --job-name "O16_DOWNSTREAM"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 pretrain_model.py
