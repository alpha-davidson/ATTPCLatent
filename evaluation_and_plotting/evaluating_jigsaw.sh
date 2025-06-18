#!/bin/bash
#SBATCH --job-name "JIGSAW_EVAL"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Adapt the models sub-folder as needed to the correct file name. Also, to plot the histogram, make sure the name of the sub-folder matches up between the models and plots folder
PYTHONPATH=../.. python3 -m ATTPCLatent.evaluation_and_plotting.evaluate_jigsaw_reconstruction --beam O16 --num-classes 24 ../relative/path/to/model ../relative/path/to/model/weight ../relative/path/to/data