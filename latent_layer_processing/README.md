# Latent Layer Analysis

This folder contains tools for exploring latent representations with
clustering/embedding methods such as t-SNE, UMAP, and k-means. It also contains
linear probing and SVM classification utilities.

## 1. Global Feature Exploration & Clustering

`global_feature_exploration.ipynb` is the interactive workspace used to explore
the geometric distribution of the latent space. It interfaces with
`clustering.py`, which implements t-SNE, UMAP, and k-means clustering,
generating projections in both 2D and 3D spaces.

The notebook preserves raw learned embeddings by default. Optional scaling is
available for distance-based visualizations when each embedding coordinate
should contribute equally. t-SNE applies an intermediate linear reduction before
the final 2D/3D embedding: PCA for dense latent arrays, or TruncatedSVD only for
genuinely sparse input matrices.

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

- SVM classification through `svm/svm.py`
- k-means clustering through `global_feature_exploration/clustering.py`
- t-SNE through `global_feature_exploration/clustering.py`

Example:

```bash
python latent_layer_processing/latent_pipeline.py \
  --name synthetic_class \
  --features data/synthetic_latent/synthetic_class_features.npy \
  --labels data/synthetic_latent/synthetic_class_labels.npy \
  --samples 50 \
  --output-dir data/synthetic_latent/class_results
```

## 3. Linear Probing

`linear_probing.py` applies a fast linear classifier over frozen latent spaces
to calculate overall classification accuracy. This evaluates how explicitly the
encoder separates fundamental physics event topologies.

Linear probing is intentionally different from exploratory visualization: it
uses the full frozen embedding vector, then fits `StandardScaler` only on the
training split before applying the same transform to the test split. PCA, UMAP,
and t-SNE are not used before the default probe.

### How to Run

Modify `linear_probing.sh` to specify the path to your target `.npy` feature
matrix, labels, and model identification name. Execute the shell script to run
the evaluation loop:

```bash
bash linear_probing.sh
```

## 4. SVM Classification & Sample Scaling

Simple probes of the latent embeddings from a pretrained model. An SVM is
trained on the frozen embeddings across increasing training sizes to test
whether similar physics events are grouped together in the learned space.

Run `svm/svm.py` with a feature file and a label file:

```bash
python latent_layer_processing/svm/svm.py \
  data/synthetic_latent/synthetic_class_features.npy \
  data/synthetic_latent/synthetic_class_labels.npy \
  --samples 50 \
  --output-dir data/synthetic_latent/class_results/svm
```

The feature file should contain an array shaped like `(num_events, latent_dim)`.
The label file should contain one class label per event. The script balances the
classes using `--samples`, trains a linear SVM, prints accuracy/F1, saves
predicted labels, and writes a confusion matrix to the output directory.
