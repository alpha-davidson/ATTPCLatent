## Mg22 Data Pre-Processing

### Input
- `output_digi_HDF_Mg22_Ne20pp_8MeV.h5`: .h5 data files use a Hierarchical Data Format (HDF), which follows a tree-like structure. Each of the highest most nodes are called keys, and these are different categories for the data. This file contains all of the point clouds for each of the events. original_keys is the list of different events. This data file has 10000 events.

### Description
- Run the `run_Mg22_voxel_pipeline.sh` on terminal to run the `Mg22_voxel_pipeline.py` script.

 - More information about this pipeline is given within the python file. You need to create `voxel_data` to insert the input file there. The `data_splits` folder will be generated containing training, test, and validation datasets for later use. 