#!/bin/bash
#SBATCH --job-name "Convert_Exp_Data"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Calling this in the terminal will run the file 'data_extraction.py.' Also, update the name of the file from which the data will be extracted.
python3 data_extraction.py name_of_the_file