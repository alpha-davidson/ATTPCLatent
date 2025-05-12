## Mg22 Data Pre-Processing

### Input
- `output_digi_HDF_Mg22_Ne20pp_8MeV.h5`: .h5 data files use a Hierarchical Data Format (HDF), which follows a tree-like structure. Each of the highest most nodes are called keys, and these are different categories for the data. This file contains all of the point clouds for each of the events. original_keys is the list of different events. This data file has 10000 events.

### Description
- Run the `Mg22_Voxel_Pipeline` notebook. Following the instructions inside closely, you will create a `voxel_data` folder, which will contain training, test, and validation datasets for later use. Running the notebook will also generate histograms of the training, test, and validation datasets post-voxelization and shuffling.