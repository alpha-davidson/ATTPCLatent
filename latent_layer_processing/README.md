Before running the script, modify the `extract_latent_layer.sh` shell script to specify the beam, num-classes, epoch, etc. Submit the shell script to extract latent space features using the `global_features_extraction.py` Python script. `clustering.py` is the main execution script called in `global_features_extraction.py`.

- ```clustering.py``` implements t-SNE and k-means clusterings, generating plots in 2- and 3-dimensional spaces.

The `clustering_plots` folder will be generated, containing information about the called clustering.