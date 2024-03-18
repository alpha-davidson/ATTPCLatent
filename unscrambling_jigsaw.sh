#!/bin/bash
#SBATCH --job-name "Jigsaw"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --output=jigsaw-%x.%j.out
#SBATCH --error=jigsaw-%x.%j.err

source /opt/conda/bin/activate tpcnet
python pretrain_on_jigsaw_events.py --num-classes 27 --num-epochs 100 voxel_data/Mg22_size512
# python ../test_gpu.py
