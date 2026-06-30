# Latent Layer Analysis

This folder contains tools for exploring latent representations with
clustering/embedding methods such as t-SNE, UMAP, and k-means. It also contains
linear probing utilities.

## 1. Global Feature Exploration & Clustering

The notebook and `clustering.py` support class-colored plots by default. For
continuous metadata such as energy or charge, pass `color_mode='continuous'` and
`values=<array>` to `UMAP_embedding` or `t_SNE_clustering`.

### How to Use

1. Ensure your model's extracted representation matrix is saved in the
   repository, such as `data/your_model_features.npy`.
2. Open `global_feature_exploration.ipynb` using your `attpc-latent` kernel.
3. Update the data loading paths to point to your target features and aligned
   labels.
4. Run the evaluation cells to compute embeddings and save results to the
   generated `plots/` folder.

The `plots` folder will be generated, containing clustering results such as
t-SNE, UMAP, and k-means plots. Plot filenames include the dataset name and
method when a `plot_name` is provided.

## 2. Latent Pipeline

`latent_pipeline.py` runs a small suite of checks on any saved latent feature
dataset:

- k-means clustering through `global_feature_exploration/clustering.py`
- t-SNE through `global_feature_exploration/clustering.py`

Example:

```bash
python latent_layer_processing/latent_pipeline.py \
  --name synthetic_class \
  --features data/synthetic_latent/synthetic_class_features.npy \
  --labels data/synthetic_latent/synthetic_class_labels.npy \
  --output-dir data/synthetic_latent/class_results
```

## 3. Linear Probing

`linear_probing.py` applies a fast linear classifier or Ridge regressor over frozen
latent spaces. Use `--task classification` (default) for class labels, or
`--task regression` for continuous targets with shape `(N,)`.

### How to Run

Modify `global_feature_exploration/linear_probing.sh` to specify the path to
your target `.npy` feature matrix, labels, and model identification name. Use
`--classifier linear-svm` for a linear SVM instead of logistic regression.
Execute the shell script from that folder:

```bash
cd global_feature_exploration
bash linear_probing.sh
```
