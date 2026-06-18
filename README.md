# AT-TPC Latent Space Evaluation

A toolkit for evaluating the quality of latent representations learned from
AT-TPC 4D point cloud data.

This repo does not handle model training or raw data preprocessing. It takes
pre-extracted embeddings as input (`.npy` files) and provides a standard set of
analysis tools that any team member can run regardless of what architecture or
framework they used to generate those embeddings.

## Input Format

Every analysis script in this repo expects two aligned `.npy` files:

| File | Shape | Description |
| --- | --- | --- |
| `embeddings.npy` | `(N, D)` | One event per row |
| `labels.npy` | `(N,)` | Integer class labels corresponding to each event |

Where `N` is the number of events and `D` is the embedding dimension. Row order
must match between the two files.

To export embeddings from your model, add the following to your evaluation
notebook:

```python
np.save("embeddings.npy", all_embeddings)
np.save("labels.npy", all_labels)
```

These `.npy` files should be saved within the `data/` directory.

## Environment

Create the repo environment from `environment.yml`:

```bash
./environment.sh
conda activate attpc-latent
```

`environment.sh` creates a fresh environment named `attpc-latent`. It does not
update an existing environment. To recreate the environment:

```bash
conda env remove -n attpc-latent
./environment.sh
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
  --samples 50 \
  --output-dir data/synthetic_latent/class_results
```

## Analysis Methods

### Unsupervised Visualization

**UMAP** and **t-SNE** project the high-dimensional embedding space down to 2D
so you can visually inspect whether the model has learned to group similar
events together without using any labels. Well-separated clusters are a good
sign that the encoder has learned meaningful structure.

### Linear Probing

A logistic regression model or linear SVM trained on top of frozen embeddings.
If a simple linear boundary can classify events accurately, it means the
relevant physics information is cleanly and explicitly encoded in the latent
space. This is the standard benchmark for representation quality.

To use linear probing:

1. Place files within the `data/` directory.
2. Change the Click parameters within `linear_probing.sh` to match the file
   path for your `.npy` files.
3. Change any other Click parameters as needed.
4. Run `linear_probing.sh` using `sbatch` or `bash` in the terminal.

### SVM Classification

A more general SVM evaluation with F1 score reporting across varying numbers of
training samples. Useful for understanding how label-efficient your
representations are.

### K-Means Clustering

Unsupervised clustering directly in embedding space, evaluated against ground
truth labels to measure how naturally the model separates event categories
without supervision.

### PCA Analysis

Uses the principal components of a given latent space to create a
lower-dimensional representation. This representation maps the data along new,
flat axes that capture the maximum variation and spread of your embeddings.
