"""Run basic checks on any saved latent-feature dataset."""

import os
import subprocess
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
SVM_SCRIPT = REPO_ROOT / "latent_layer_processing" / "svm" / "svm.py"

sys.path.insert(0, str(CLUSTERING_DIR))
from clustering import k_means_clustering, t_SNE_clustering  # noqa: E402


def run_svm(features, labels, samples, output_dir):
    command = [
        sys.executable,
        str(SVM_SCRIPT),
        str(features),
        str(labels),
        "--samples",
        str(samples),
        "--output-dir",
        str(output_dir / "svm"),
    ]
    subprocess.run(command, check=True)


@click.command()
@click.option("--name", required=True, help="Short name printed in results.")
@click.option("--features", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--labels", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--samples", default=50, type=click.INT, help="Number of examples per class for SVM training.")
@click.option(
    "--output-dir",
    default=Path("latent_pipeline_smoke_results"),
    type=click.Path(path_type=Path),
    help="Directory for outputs such as SVM confusion matrices.",
)
@click.option("--perplexity", default=20, type=click.INT, help="t-SNE perplexity.")
def main(name, features, labels, samples, output_dir, perplexity):
    feature_data = np.load(features)
    label_data = np.load(labels)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_svm(features, labels, samples, output_dir)

    # The clustering helpers save plots relative to their own folder.
    os.chdir(CLUSTERING_DIR)
    _, cluster_labels, _ = k_means_clustering(
        feature_data,
        label_data,
        dimension=2,
        save_dir=None,
        num_samples_to_print=3,
        plot_name=name,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    tsne = t_SNE_clustering(
        feature_data,
        dimension=2,
        ax=ax,
        color="tab:blue",
        label=name,
        alpha=0.7,
        perplexity=perplexity,
        plot_name=name,
    )
    plt.close(fig)

    print(f"\n{name}")
    print(f"  features: {feature_data.shape}")
    print(f"  k-means adjusted rand: {adjusted_rand_score(label_data, cluster_labels):.4f}")
    print(f"  t-SNE output: {tsne.shape}")


if __name__ == "__main__":
    main()
