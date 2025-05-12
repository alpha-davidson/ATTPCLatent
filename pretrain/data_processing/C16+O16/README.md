## C16+O16 Data Pre-Processing

### Input
- `O16_run160.h5`
- `example_run_0017_clusters.h5`
- `example_run_0017_fit_2H.parquet`

### Description
- Run the `C16+O16_voxel_pipeline` notebook. Following the instructions inside closely, you will create `voxel_data` and `data_splits` folders, which will contain training, test, and validation datasets for later use. Running the notebook will also generate histograms of the training, test, and validation datasets post-voxelization and shuffling.