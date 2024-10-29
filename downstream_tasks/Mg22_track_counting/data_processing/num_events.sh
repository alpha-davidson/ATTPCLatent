#!/bin/bash
#SBATCH --job-name "Mg22_DW_NUM_EVENTS_SPLIT"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 num_events_pipeline.py --count-023 700 --count-1 59 --count-4 11 --count-5 4 --iterations 10
