#!/bin/bash
#SBATCH --job-name "Convert_Sim_Data"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Calling this in the terminal will run the file 'data_extraction.py.' Also, update the name of the file from which the data will be extracted.
python3 data_extraction.py --beam Mg22 output_digi_HDF_Mg22_Ne20pp_8MeV