#!/bin/bash
#SBATCH --job-name "O16_DOWNSTREAM"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 Mg22_dw_task.py
