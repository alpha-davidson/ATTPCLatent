# AT-TPC Latent Space Evaluation

A toolkit for evaluating the quality of latent representations learned from
AT-TPC 4D point cloud data.

This repo does not handle model training or raw data preprocessing. It takes
pre-extracted embeddings as input (`.npy` files) and provides a standard set of
analysis tools that any team member can run regardless of what architecture or
framework they used to generate those embeddings.

## Input Format

Every analysis script expects a feature matrix and a matching label vector:

| File | Shape | Description |
| --- | --- | --- |
| `embeddings.npy` | `(N, D)` | One event per row |
| `labels.npy` | `(N,)` | Class label per event |

Where `N` is the number of events and `D` is the embedding dimension. Row order
must match between the two files.

**Partially labeled data:** assign unlabeled events a dedicated label value such
as `-1`. That value is treated as its own class in every analysis (plots, k-means,
linear probing), not filtered out. On label-colored plots it is always drawn in
grey so it stands out from verified classes. Use `class_names` in the notebook or
`--class-name` in linear probing to give it a readable legend name.

To export embeddings from your model, add the following to your evaluation
notebook:

```python
np.save("embeddings.npy", all_embeddings)
np.save("labels.npy", all_labels)
```

These `.npy` files should be saved within the `data/` directory.

## Environment

From the repository root, create the environment from
`environment.yml`:

```bash
./environment.sh
conda activate attpc-latent
```

`environment.sh` creates a fresh environment named `attpc-latent` and registers
a matching Jupyter kernel (`attpc-latent`). It does not update an existing
environment. To recreate the environment:

```bash
conda env remove -n attpc-latent
./environment.sh
```

If you already created the environment before kernel registration was added,
register the kernel manually:

```bash
conda activate attpc-latent
python -m ipykernel install --user --name attpc-latent --display-name "attpc-latent"
```

## Synthetic Latent Data

The `synthetic_data` folder can generate small local `.npy` files for testing
the latent-processing tools:

```bash
python synthetic_data/generate_synthetic_latent_data.py
```

This writes ignored local files to `data/synthetic_latent/`. To run the generic
latent pipeline on one of those datasets:

```bash
python latent_layer_processing/latent_pipeline.py \
  --name synthetic_class \
  --features data/synthetic_latent/synthetic_class_features.npy \
  --labels data/synthetic_latent/synthetic_class_labels.npy \
  --output-dir data/synthetic_latent/class_results
```

## Analysis Methods

### Unsupervised Visualization

**UMAP** and **t-SNE** project the high-dimensional embedding space down to 2D
so you can visually inspect whether the model has learned to group similar
events together. Points are colored by label; unlabeled events (default `-1`)
appear in grey.
Well-separated clusters are a good sign that the encoder has learned meaningful
structure.

### Linear Probing

A logistic regression model or linear SVM trained on top of frozen embeddings.
If a simple linear boundary can classify events accurately, it means the relevant
physics information is cleanly and explicitly encoded in the latent space. This
is the standard benchmark for representation quality. Every label value in the
file is treated as a class, including unlabeled sentinels such as `-1`.

To use linear probing:

1. Place files within the `data/` directory.
2. Change the Click parameters within
   `latent_layer_processing/global_feature_exploration/linear_probing.sh` to
   match the file path for your `.npy` files.
3. Change any other Click parameters as needed (for example, `--classifier
   linear-svm`).
4. From `latent_layer_processing/global_feature_exploration/`, run
   `linear_probing.sh` using `sbatch`.

### K-Means Clustering

Unsupervised clustering directly in embedding space, evaluated against the
provided labels. The number of clusters matches the number of unique label
values, including any unlabeled group.

### PCA Analysis

Uses the principal components of a given latent space to create a
lower-dimensional representation. This representation maps the data along new,
flat axes that capture the maximum variation and spread of your embeddings.
PCA variance analysis runs without label coloring; labeled projection plots use
every class in the label file.
