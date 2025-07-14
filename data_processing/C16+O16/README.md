## C16+O16 Data Pre-Processing

### Input
- `O16_run160.h5`
- `example_run_0017_clusters.h5`
- `example_run_0017_fit_2H.parquet`

### Description
- Run the `C16+O16_data_pipeline.sh` on terminal to run the `C16+O16_data_pipeline.py` script.

 - More information about this pipeline is given within the python file. You need to create `voxel_data` to insert the input file there. The `data_splits` folder will be generated containing training, test, and validation datasets for later use. 