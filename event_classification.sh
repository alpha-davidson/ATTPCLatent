#!/bin/bash
#SBATCH --job-name "Event-Classification"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --exclude=phy1kuchera

python3 train_event_classification_model.py --num-classes 6 data/Mg22_size512_convertXYZ

# python3 evaluate_pointnet.py $SLURM_JOB_ID $type $sem_seg $num_classes $voxel_shuffle