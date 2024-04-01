#!/bin/bash
#SBATCH --job-name "full save mg22 shuffled"
#SBATCH --mem 32G
#SBATCH --gpus 1

<<<<<<< HEAD
source /opt/conda/bin/activate tpcnet
python pretrain_on_jigsaw_events.py --num-classes 27 --num-epochs 200 voxel_data/Mg22_size512
# python ../test_gpu.py
=======

python3 pretrain_on_jigsaw_events.py --num-classes 27 --num-epochs 5 voxel_data/Mg22_size512

>>>>>>> be5dbb4bbdaee2e4c51f81e2dfd116bc54a4cbf3
