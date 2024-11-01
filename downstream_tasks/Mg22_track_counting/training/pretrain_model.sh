#!/bin/bash
#SBATCH --job-name "Mg22_DOWNSTREAM"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 pretrain_model.py --model-num 2174
