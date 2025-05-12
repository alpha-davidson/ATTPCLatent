## C16 Data Pre-Processing

### Input
- `example_run_0017_clusters.h5`
- `example_run_0017_fit_2H.parquet`

### Description
- Run the `C16_data_pipeline.sh` on terminal to run the `C16_data_pipeline_class.py` script, which convert the h5 file to a numpy file that the `C16_pipeline.ipynb` data processing pipeline can take in.

 - More information about this pipeline is given within the python file and the notebook. You will create `voxel_data` and `data_splits` folders, which will contain training, test, and validation datasets for later use. Running the notebook will also generate histograms of the training, test, and validation datasets post-voxelization and shuffling.