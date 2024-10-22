#!/bin/bash
#SBATCH --job-name "train_model_class"
#SBATCH --mem 32G
#SBATCH --gpus 1

#source /opt/conda/bin/activate pointnet

#C16 model:
#python test_pointnet.py --model-num X --data-folder C16_data --file-extension C16_size256_

#Mg22 model:
#python test_pointnet.py --model-num X --data-folder Mg22_data --file-extension Mg22_size512_

#For Classification: for now, classification doesn't need anything
python pointnet_classification.py --model-num 49 --data-folder Mg22_dw_data --file-extension Mg22_size512_