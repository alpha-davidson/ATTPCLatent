#!/bin/bash
#SBATCH --job-name "Mg22_DW_NUM_EVENTS_SPLIT"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 Mg22_dw_num_events_pipeline.py
