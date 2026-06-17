# Latent Layer Analysis

## 1. Global Feature Exploration & Clustering

`global_feature_exploration.ipynb` is the interactive workspace used to explore the geometric distribution of the latent space. It interfaces with `clustering.py`, which implements **t-SNE**, **UMAP**, and **k-means** clustering, generating projections in both 2D and 3D spaces.

### How to Use:
1. Ensure your model's extracted representation matrix is dropped into the repository (e.g., `data/your_model_features.npy`).
2. Open `global_feature_exploration.ipynb` using your `attpc-eval` kernel.
3. Update the data loading paths to point to your target features and the aligned `labels/master_labels.npy`.
4. Run the evaluation cells to compute embeddings and automatically save results to the generated `plots/` folder.

---

## 2. Linear Probing

`linear_probing.py` applies a fast, deterministic linear classifier (Logistic Regression / Linear SVM) over the frozen latent spaces to calculate overall classification accuracy. This evaluates how explicitly the encoder separates fundamental physics event topologies (e.g., 2-track vs. 3-track).

### How to Run:
Modify the `linear_probing.sh` shell script to specify the path to your target `.npy` feature matrix, the master labels, and the model identification name. Execute the shell script to run the evaluation loop:

```bash
bash linear_probing.sh
```

---


## 3. SVM Classification & Sample Scaling
Simple probes of the latent embeddings from a pretrained model. An SVM is trained on the frozen embeddings across increasing training sizes to test whether similar physics events are grouped together in the learned space.

svm.py fits a Support Vector Machine to the data, which can be dynamically profiled using plot.py to map downstream F1-scores relative to the number of available training samples.

### How to Run:
Modify `plot.sh` to specify your target input files and the desired sample size ranges. Run the script to generate the curves:

```bash
bash plot.sh
```

---


