#!/bin/bash
#SBATCH --job-name "PointNet"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --exclude=phy1kuchera

type='BINARY'
sem_seg=True
voxel_shuffle=True
num_classes=27
mkdir $SLURM_JOB_ID

python3 train_pointnet.py $sem_seg $num_classes $voxel_shuffle
python3 evaluate_pointnet.py $SLURM_JOB_ID $type $sem_seg $num_classes $voxel_shuffle