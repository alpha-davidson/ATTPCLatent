"""Run basic checks on any saved latent-feature dataset."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import click
import numpy as np
from sklearn.metrics import adjusted_rand_score


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTERING_DIR = REPO_ROOT / "latent_layer_processing" / "global_feature_exploration"

sys.path.insert(0, str(CLUSTERING_DIR))
from clustering import k_means_clustering, pca_variance_analysis, t_SNE_clustering  # noqa: E402
from label_utils import labeled_mask, require_labeled_rows  # noqa: E402


@click.command()
@click.option("--name", required=True, help="Short name printed in results.")
@click.option("--features", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--labels", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    default=Path("latent_pipeline_smoke_results"),
    type=click.Path(path_type=Path),
    help="Directory for k-means plots and other pipeline outputs.",
)
@click.option("--perplexity", default=20, type=click.INT, help="t-SNE perplexity.")
def main(name, features, labels, output_dir, perplexity):
    feature_data = np.load(features)
    label_data = np.load(labels).ravel()
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_variance_analysis(
        feature_data,
        save_dir=str(output_dir),
        plot_name=name,
    )

    labeled_mask_array = labeled_mask(label_data)
    if labeled_mask_array.any():
        require_labeled_rows(label_data, context="Label-dependent pipeline steps")

        os.chdir(CLUSTERING_DIR)
        _, cluster_labels, _ = k_means_clustering(
            feature_data,
            label_data,
            dimension=2,
            save_dir=str(output_dir),
            num_samples_to_print=3,
            plot_name=name,
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        tsne = t_SNE_clustering(
            feature_data,
            dimension=2,
            ax=ax,
            labels=label_data,
            perplexity=perplexity,
            plot_name=name,
            save_dir=str(output_dir),
        )
        plt.close(fig)

        ari = adjusted_rand_score(label_data[labeled_mask_array], cluster_labels[labeled_mask_array])
        print(f"\n{name}")
        print(f"  features: {feature_data.shape}")
        print(f"  k-means adjusted rand (labeled rows only): {ari:.4f}")
        print(f"  t-SNE output: {tsne.shape}")
    else:
        print(f"\n{name}")
        print(f"  features: {feature_data.shape}")
        print("  skipped k-means and t-SNE (no labeled rows found)")


if __name__ == "__main__":
    main()
