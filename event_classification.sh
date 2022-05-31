#!/bin/bash
#SBATCH --job-name "Event-Classification"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --exclude=phy1kuchera

python3 train_event_classification_model.py data/Mg22_size512train_convertXYZ.npy data/Mg22_size512val_convertXYZ.npy 
# python3 evaluate_pointnet.py $SLURM_JOB_ID $type $sem_seg $num_classes $voxel_shuffle