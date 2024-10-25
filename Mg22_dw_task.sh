#!/bin/bash
#SBATCH --job-name "Mg22_DOWNSTREAM"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 Mg22_dw_task.py --model-num 63
