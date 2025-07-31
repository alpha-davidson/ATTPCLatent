#!/bin/bash
#SBATCH --job-name "C16_PIPELINE"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 C16_data_pipeline_merge.py
