## Processing Raw Mg22 Data
Create `data` folder and put `output_digi_HDF_Mg22_Ne20pp_8MeV.h5` inside the folder.
Run the `data_pipeline.ipynb` notebook, which will convert the original h5 file to npy file, and generate training, test, and validation datasets for later use. All the generated files can be found in the data folder.
## Creating Smaller Training Sets
Modify the `num_events.sh` file to change the target number of events. 
Remember to check the total number of events for each track count after re-running the `data_pipeline.ipynb` notebook. The numbers will change every time you run the pipeline. This information can be found near the end of the pipeline, at the console right before "Split into features and labels."
After modifying the target number of events, sbatch the `num_events.sh` file. This file will generate two folders inside the data folder: `Mg22_size512_{target_number}train_features` and `Mg22_size512_{target_number}train_labels`. Inside each folder, you will see 10 npy files, each corresponding to one iteration of the sampling. These files will be used for training.