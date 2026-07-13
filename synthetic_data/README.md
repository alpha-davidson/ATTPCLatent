# Synthetic latent data

This folder contains a simple script for making fake latent-feature arrays. The
arrays are example inputs for the generic latent-processing smoke test in
`latent_layer_processing/latent_pipeline.py`.

Generated `.npy` files are written to `data/synthetic_latent/`. They are
local-only test artifacts and should not be committed to GitHub.

## Scripts

- `generate_synthetic_latent_data.py`: Creates the synthetic arrays.

## Run

From the repository root, first create local synthetic data:

```bash
python synthetic_data/generate_synthetic_latent_data.py
```

Then run the generic latent smoke test on each dataset:

```bash
python latent_layer_processing/latent_pipeline.py \
  --name synthetic_class \
  --features data/synthetic_latent/synthetic_class_features.npy \
  --labels data/synthetic_latent/synthetic_class_labels.npy \
  --output-dir data/synthetic_latent/class_results

python latent_layer_processing/latent_pipeline.py \
  --name synthetic_gaussian_noise \
  --features data/synthetic_latent/synthetic_gaussian_noise_features.npy \
  --labels data/synthetic_latent/synthetic_gaussian_noise_labels.npy \
  --output-dir data/synthetic_latent/gaussian_noise_results
```

The generator writes two datasets:

- Gaussian noise features with random labels. This is a negative control.
- Class-structured features. Each class is centered on a different latent
  dimension, so a simple classifier should separate the classes easily.

No random seed is set. Re-running these scripts should show natural variability
in both the generated data and algorithms such as k-means and t-SNE.
