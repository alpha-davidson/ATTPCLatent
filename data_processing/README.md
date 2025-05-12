This folder currently contains data pipelines for four different datasets and an event-wise voxel data exploration notebook. Choose the dataset and the corresponding workflow guide is inside its folder.

## Explanation of Voxels Shuffling
Voxel shuffling is an essential part of this pre-training model, and it is achieved by running the data pipeline.

### Creating Voxel Data
Voxel datasets are compiled using the `[experiment]_Voxel_pipeline.ipynb` file. The data file from the TPC is first loaded, and points are randomly sampled (NOT a completely randomized process, however - random samples are taken according to the event length desired).  
In this stage, points are first normalized into a unit cube, segmented into voxels (K_x x K_y x K_z cube), and then assigned labels. Labeled voxels are then shuffled such that each voxel is assigned an ID other than its own. Points are then moved to their new voxel, and the current version of the notebook randomly augments points for better xyz generalization. It is then checked that the boundaries of the unit cube have not been violated. 
The files of shuffled voxels are split into training, validation, and test sets, which are saved in the `voxel_data` folder. Lastly, these datasets are checked for NaNs and infs, and a histogram is created.

### Voxel Orientation
Each voxel will be assigned an integer starting from 0 up to (K_x x K_y x K_z) - 1. Voxel number 0 has a bottom corner at the origin and the top, opposite corner at x = 1/K_x, y = 1/K_y, z = 1/K_z. The next voxel, voxel 1, has the same y and z coordinates as voxel 0 while the x coordinate moves forward. In the current state of the repository, K_x = 2, K_y = 2, and K_z = 6.
