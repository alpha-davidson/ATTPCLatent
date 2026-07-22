import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
from skdim.id import TwoNN

from utils import (
    DEFAULT_UNLABELED_VALUES,
    validate_features_and_labels,
    validate_plot_dimension,
    get_class_names,
    get_plot_colors,
    plot_labeled_embedding,
)

def t_SNE_clustering(features, dimension, ax, labels, perplexity, class_names=None,
                     plt_colors=None, save_dir=None, plot_name=None, random_state=None,
                     unlabeled_values=DEFAULT_UNLABELED_VALUES):
    """Run t-SNE and plot points colored by class label.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    plt_colors is optional; when omitted, one color is generated per class.
    Rows matching unlabeled_values are always drawn in grey.
    """

    validate_plot_dimension(dimension, "t_SNE_clustering")
    features, labels = validate_features_and_labels(features, labels)
    labels, unique_labels, class_names = get_class_names(labels, class_names)

    # create a folder for t-SNE clustering
    folder_path = os.path.join(save_dir or "../plots", "t-sne", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f"t_sne_{dimension}d"

    # apply t-SNE clustering over all features at once
    model = TSNE(n_components=dimension, perplexity=perplexity, random_state=random_state)
    tsne_data = model.fit_transform(features)

    plot_labeled_embedding(
        tsne_data, labels, unique_labels, class_names, ax, plt_colors,
        unlabeled_values=unlabeled_values,
    )
    ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_perplexity_{perplexity}.png"), dpi=200)

    return tsne_data

def UMAP_embedding(features, dimension, ax, labels, neighbors, class_names=None,
                   plt_colors=None, save_dir=None, plot_name=None, random_state=None,
                   unlabeled_values=DEFAULT_UNLABELED_VALUES):
    """Run UMAP and plot points colored by class label.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    plt_colors is optional; when omitted, one color is generated per class.
    Rows matching unlabeled_values are always drawn in grey.
    """
    validate_plot_dimension(dimension, "UMAP_embedding")
    features, labels = validate_features_and_labels(features, labels)
    labels, unique_labels, class_names = get_class_names(labels, class_names)

    # create a folder for UMAP embedding
    folder_path = os.path.join(save_dir or "../plots", "umap", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f"umap_{dimension}d"

    # apply UMAP embedding over all features at once
    model = umap.UMAP(n_components=dimension, n_neighbors=neighbors, random_state=random_state)
    umap_data = model.fit_transform(features)

    plot_labeled_embedding(
        umap_data, labels, unique_labels, class_names, ax, plt_colors,
        unlabeled_values=unlabeled_values,
    )
    ax.figure.savefig(os.path.join(folder_path, f"{plot_name}_neighbors_{neighbors}.png"), dpi=200)

    return umap_data

def k_means_clustering(features, labels, dimension, save_dir=None, num_samples_to_print=10,
                       plot_name=None, class_names=None, random_state=None,
                       unlabeled_values=DEFAULT_UNLABELED_VALUES):
    """Run k-means and compare cluster assignments with known labels.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    Rows matching unlabeled_values are always drawn in grey on the true-label panel.
    """
    validate_plot_dimension(dimension, "k_means_clustering")

    features, labels = validate_features_and_labels(features, labels)
    folder_path = os.path.join(save_dir or "../plots/k_means", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    if plot_name is None:
        plot_name = f'k_means_{dimension}d'

    # k-means clustering on full feature space
    labels, unique_labels, class_names = get_class_names(labels, class_names)
    k = len(unique_labels)
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init='auto', random_state=random_state)
    cluster_labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    print("\nRandom sample indices from each cluster:")
    indices = []
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        random_indices = np.random.choice(cluster_indices, 
                                          size=min(num_samples_to_print, len(cluster_indices)), 
                                          replace=False)
        indices.append(random_indices)
        print(f"Cluster {cluster_id}: {random_indices.tolist()}")

    pca = PCA(n_components=dimension)
    reduced_features = pca.fit_transform(features)
    cluster_names = [f"Cluster {i}" for i in range(k)]
    cluster_values = np.arange(k)
    centroid_features = pca.transform(centroids)

    if dimension == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plot_labeled_embedding(reduced_features, cluster_labels, cluster_values, cluster_names, ax1, size=5, alpha=1.0)
        ax1.scatter(centroid_features[:, 0], centroid_features[:, 1],
                    marker='x', s=60, c='black', label='Centroids')
        ax1.set_title("KMeans Clustering")
        ax1.legend()

        plot_labeled_embedding(
            reduced_features, labels, unique_labels, class_names, ax2, size=5, alpha=1.0,
            unlabeled_values=unlabeled_values,
        )
        ax2.set_title("True Labels")
        ax2.legend()

    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        plot_labeled_embedding(reduced_features, cluster_labels, cluster_values, cluster_names, ax1, size=5, alpha=1.0)
        ax1.scatter(*centroid_features.T, marker='x', s=60, c='black', label='Centroids')
        ax1.set_title("KMeans Clustering")
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        plot_labeled_embedding(
            reduced_features, labels, unique_labels, class_names, ax2, size=5, alpha=1.0,
            unlabeled_values=unlabeled_values,
        )
        ax2.set_title("True Labels")
        ax2.legend()

    plt.tight_layout()
    filename = f"kmeans_{dimension}d.png" if plot_name is None else f"{plot_name}_kmeans_{dimension}d.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=200)
    plt.close()

    return features, cluster_labels, indices

def pca_clustering(features, labels, dimension, save_dir=None, class_names=None,
                   plot_name=None, random_state=None,
                   unlabeled_values=DEFAULT_UNLABELED_VALUES):
    """Project features with PCA and plot points colored by class label.

    labels must contain one label for each row in features. class_names is
    optional; when provided as a list, it follows np.unique(labels) order.
    Rows matching unlabeled_values are always drawn in grey.
    """
    validate_plot_dimension(dimension, "pca_clustering")

    features, labels = validate_features_and_labels(features, labels)
    folder_path = os.path.join(save_dir or "../plots/pca", f"{dimension}d_plots")
    os.makedirs(folder_path, exist_ok=True)

    n_components = min(dimension, features.shape[0], features.shape[1])
    if n_components != dimension:
        warnings.warn(
            f"Requested {dimension} PCA components, but only {n_components} are possible "
            f"for features with shape {features.shape}.",
            UserWarning,
        )

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(features)

    labels, unique_labels, class_names = get_class_names(labels, class_names)

    if dimension == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_labeled_embedding(
            reduced_features, labels, unique_labels, class_names, ax, size=5, alpha=1.0,
            unlabeled_values=unlabeled_values,
        )
        ax.set_title("PCA Projection")
        ax.legend()
        filename = "pca_2d.png" if plot_name is None else f"{plot_name}_pca_2d.png"
        plt.savefig(os.path.join(folder_path, filename), dpi=200)
        plt.close()

    elif dimension == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_labeled_embedding(
            reduced_features, labels, unique_labels, class_names, ax, size=5, alpha=1.0,
            unlabeled_values=unlabeled_values,
        )
        ax.set_title("PCA Projection")
        ax.legend()
        filename = "pca_3d.png" if plot_name is None else f"{plot_name}_pca_3d.png"
        plt.savefig(os.path.join(folder_path, filename), dpi=200)
        plt.close()

    return reduced_features


def pca_variance_analysis(features, variance_threshold=0.95, save_dir=None,
                          plot_name=None, random_state=None):
    folder_path = os.path.join(save_dir or "../plots/pca", "variance")
    os.makedirs(folder_path, exist_ok=True)

    n_components = min(features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(features)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_needed = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    variance_at_threshold = float(cumulative_variance[n_needed - 1])

    print(
        f"\nPCA variance analysis: {n_needed} component(s) explain "
        f"{variance_at_threshold:.1%} of variance (threshold={variance_threshold:.0%})"
    )

    plt.figure(figsize=(7, 5))
    component_numbers = np.arange(1, n_components + 1)
    plt.plot(component_numbers, cumulative_variance, marker="o", markersize=3)
    plt.axhline(variance_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(n_needed, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = "cumulative_variance.png" if plot_name is None else f"{plot_name}_cumulative_variance.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=200)
    plt.close()

    return {
        "n_components": n_needed,
        "variance_threshold": variance_threshold,
        "variance_explained": variance_at_threshold,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
    }


def _fit_twonn_dimension(features, discard_fraction=0.1):
    """Return a single TwoNN intrinsic dimension estimate for one point cloud."""
    estimator = TwoNN(discard_fraction=discard_fraction)
    estimator.fit(features)
    return float(estimator.dimension_)


def _default_twonn_subsample_sizes(n_total, max_subsample):
    """Log-spaced subsample sizes capped by the available number of points."""
    template = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    upper_bound = min(n_total, max_subsample)
    sizes = [n for n in template if 3 <= n <= upper_bound]
    if upper_bound not in sizes and upper_bound >= 3:
        sizes.append(upper_bound)
    return sorted(set(sizes))


def _estimate_plateau_intrinsic_dimension(sample_sizes, means, stds):
    """Pick the ID from the most stable contiguous region of the scale curve."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    if len(means) == 1:
        return float(means[0])

    window = min(3, len(means))
    best_score = np.inf
    plateau_estimate = float(means[-1])
    for start in range(len(means) - window + 1):
        window_means = means[start:start + window]
        window_stds = stds[start:start + window]
        score = np.ptp(window_means) + np.mean(window_stds)
        if score < best_score:
            best_score = score
            plateau_estimate = float(np.median(window_means))
    return plateau_estimate


def twonn_intrinsic_dimension(
    features,
    discard_fraction=0.1,
    sample_sizes=None,
    n_trials=10,
    max_subsample=5000,
    max_samples=None,
    random_state=None,
    save_dir=None,
    plot_name=None,
):
    """Estimate intrinsic dimension with TwoNN over random subsamples of many sizes.

    Re-running TwoNN on smaller subsamples changes the neighbor scale and helps
    separate a stable plateau ID from noise-driven estimates at very large N.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(
            f"Expected shape (n_events, embedding_dim), got {features.shape}"
        )
    if features.shape[0] < 3:
        raise ValueError("TwoNN requires at least 3 samples.")

    n_total = features.shape[0]
    if max_samples is not None and n_total > max_samples:
        rng = np.random.default_rng(random_state)
        pool_indices = rng.choice(n_total, size=max_samples, replace=False)
        features = features[pool_indices]
        n_total = features.shape[0]

    if sample_sizes is None:
        sample_sizes = _default_twonn_subsample_sizes(n_total, max_subsample)
    else:
        sample_sizes = sorted({int(n) for n in sample_sizes if 3 <= int(n) <= n_total})

    if not sample_sizes:
        raise ValueError(
            f"No valid subsample sizes for n_total={n_total}. "
            "Provide smaller sample_sizes or increase max_subsample."
        )

    embedding_dimension = int(features.shape[1])
    rng = np.random.default_rng(random_state)
    means = []
    stds = []

    print(
        f"\nTwoNN scale-curve analysis "
        f"(embedding dimension={embedding_dimension}, pool size={n_total}, "
        f"trials per size={n_trials}, discard_fraction={discard_fraction})"
    )
    for n in sample_sizes:
        trial_estimates = []
        for trial in range(n_trials):
            trial_seed = None if random_state is None else random_state + n + trial
            trial_rng = np.random.default_rng(trial_seed)
            indices = trial_rng.choice(n_total, size=n, replace=False)
            trial_estimates.append(
                _fit_twonn_dimension(features[indices], discard_fraction=discard_fraction)
            )
        mean_d = float(np.mean(trial_estimates))
        std_d = float(np.std(trial_estimates))
        means.append(mean_d)
        stds.append(std_d)
        print(f"  N={n:5d} : ID = {mean_d:.2f} +/- {std_d:.2f}")

    intrinsic_dimension = _estimate_plateau_intrinsic_dimension(sample_sizes, means, stds)
    print(
        f"Plateau intrinsic dimension: {intrinsic_dimension:.2f} "
        f"(embedding dimension={embedding_dimension})"
    )

    folder_path = os.path.join(save_dir or "../plots/twonn", "estimates")
    os.makedirs(folder_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sample_sizes,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        color="#005aff",
        label="Estimated ID (TwoNN)",
    )
    plt.axhline(
        y=intrinsic_dimension,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Plateau ID = {intrinsic_dimension:.2f}",
    )
    plt.axhline(
        y=embedding_dimension,
        color="#ff4b00",
        linestyle=":",
        linewidth=1,
        label=f"Embedding dimension = {embedding_dimension}",
    )
    plt.xscale("log")
    plt.xlabel("Subsample size (N)")
    plt.ylabel("Intrinsic dimension (d)")
    plt.title("TwoNN Intrinsic Dimension vs Subsample Size")
    plt.xticks(sample_sizes, labels=sample_sizes)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    filename = (
        "twonn_scale_curve.png"
        if plot_name is None
        else f"{plot_name}_twonn_scale_curve.png"
    )
    plt.savefig(os.path.join(folder_path, filename), dpi=200)
    plt.close()

    return {
        "intrinsic_dimension": intrinsic_dimension,
        "embedding_dimension": embedding_dimension,
        "n_samples_total": int(n_total),
        "sample_sizes": sample_sizes,
        "means": means,
        "stds": stds,
        "n_trials": n_trials,
        "discard_fraction": discard_fraction,
        "max_subsample": max_subsample,
    }
