# AT-TPC Latent Space Evaluation

A toolkit for evaluating the quality of latent representations learned from
AT-TPC 4D point cloud data.

This repo does not handle model training or raw data preprocessing. It takes
pre-extracted embeddings as input (`.npy` files) and provides a standard set of
analysis tools that any team member can run regardless of what architecture or
framework they used to generate those embeddings.

## Input Format

Every analysis script expects a feature matrix and, for most tools, a matching
label vector:

| File | Shape | Description |
| --- | --- | --- |
| `embeddings.npy` | `(N, D)` | One event per row |
| `labels.npy` | `(N,)` | Class label per event (optional for some tools) |

Where `N` is the number of events and `D` is the embedding dimension. Row order
must match between the two files when labels are provided.

### Label policy

Use **verified reference labels** — simulation truth, manual review, or another
trusted taxonomy. **Do not** use the embedding model's own predictions as labels
for supervised benchmarks (linear probing, k-means comparison).

| Tool | Labels required? |
| --- | --- |
| PCA variance | No |
| k-means, PCA/UMAP/t-SNE plots | Optional (unlabeled rows shown in gray) |
| Linear probing | Yes (labeled rows only) |

**Partially labeled data:** mark unlabeled rows with `-1` (default). Supervised
tools ignore those rows automatically.

**Fully unlabeled data:** omit `labels.npy` in the notebook (`labels_path =
None`) or provide an all-`-1` vector. Only label-free analyses (PCA variance)
will run.

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

**UMAP** and **t-SNE** project the high-dimensional embedding space down to 2D.
When verified labels are available, points are colored by class and unlabeled
rows appear in gray. Well-separated clusters are a good sign that the encoder has
learned meaningful structure.

### Linear Probing

A logistic regression model or linear SVM trained on top of frozen embeddings
using **verified labels only**. If a simple linear boundary can classify events
accurately, it means the relevant physics information is cleanly and explicitly
encoded in the latent space. This is the standard benchmark for representation
quality. Unlabeled rows (`-1`) are ignored.

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

Unsupervised clustering directly in embedding space, evaluated against verified
labels when available. Unlabeled rows are shown in gray and excluded from the
class count used to set k.

### PCA Analysis

Uses the principal components of a given latent space to create a
lower-dimensional representation. **PCA variance analysis** works without
labels; labeled PCA/UMAP/t-SNE projections require at least one verified class.
