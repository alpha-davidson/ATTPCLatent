This folder contains the implementation of global feature extraction and exploration using different kinds of clustering/embedding, such as t-SNE, UMAP, and k-means. This folder also contains the implementation of linear probing and SVM classifier. 

# Global Feature Extraction

Before running the script, modify the `extract_latent_layer.sh` shell script to specify the beam, num-classes, etc. Submit the shell script to extract latent space features using the `global_feature_extraction.py` Python script. The `global_features` folder will be generated, containing the arrays of extracted global features.

# Global Feature Exploration

`global_feature_exploration.ipynb` is the main file used for global feature exploration. Adapt the parameters outlined in the file to get the desired results.

`clustering.py` is the main execution script called in `global_feature_exploration.ipynb`.

- ```clustering.py``` implements t-SNE, UMAP, and k-means clusterings, generating plots in 2- and 3-dimensional spaces.

The `plots` folder will be generated, containing the specified clustering results, such as t-SNE, UMAP, or k-means.

# Linear Probing

Before running the script, modify the `linear_probing.sh` shell script to specify the beam, num-classes, model, etc. Submit the shell script to apply linear probing using the `linear_probing.py` Python script. 

# SVM Classification

Modify the `svm.py` Python script to specify the name of the global feature file and its labels as well. Additionally, `svm.py` can be launched through `plot.py`, which generates a plot with f1 scores in relation to the number of samples. Specify the number of samples in `plot.py` and run it by using `plot.sh`.