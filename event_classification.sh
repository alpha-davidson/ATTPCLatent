#!/bin/bash
#SBATCH --job-name "Event-Classification"
#SBATCH --mem 32G
#SBATCH --gpus 1

python3 train_event_classification_model.py --num-classes 6 data/Mg22_size512_convertXYZ
