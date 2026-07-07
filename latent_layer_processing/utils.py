"""Shared utility functions used across latent_layer_processing scripts.

Functions here are intentionally small, dependency-free (numpy + matplotlib
only), and have no side effects. Import them from anywhere in the package:

    from utils import validate_features_and_labels, get_class_names, get_plot_colors

Naming convention
-----------------
Public functions (validate_features_and_labels, get_class_names, …) are the
canonical versions. The leading-underscore aliases at the bottom of this module
exist so that clustering.py can continue using its private-style names during
the migration period without any runtime changes.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Feature / label validation
# ---------------------------------------------------------------------------

def validate_features_and_labels(features, labels):
    """Validate that every feature row has exactly one class label.

    Parameters
    ----------
    features : array-like, shape (N, D)
    labels   : array-like, shape (N,)

    Returns
    -------
    features : np.ndarray — unchanged
    labels   : np.ndarray, 1-D
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(
            "labels must be a one-dimensional array with one label per feature row; "
            f"got shape {labels.shape}."
        )
    if len(features) != len(labels):
        raise ValueError(
            "features and labels must contain the same number of rows; "
            f"got {len(features)} features and {len(labels)} labels."
        )
    return features, labels


def validate_features_and_targets(features, targets):
    """Validate that every feature row has exactly one regression target.

    Parameters
    ----------
    features : array-like, shape (N, D)
    targets  : array-like, shape (N,) — must be finite floats

    Returns
    -------
    features : np.ndarray — unchanged
    targets  : np.ndarray, 1-D float
    """
    targets = np.asarray(targets, dtype=float).ravel()
    if targets.ndim != 1:
        raise ValueError(
            "targets must be a one-dimensional array with one value per feature row; "
            f"got shape {targets.shape}."
        )
    if len(features) != len(targets):
        raise ValueError(
            "features and targets must contain the same number of rows; "
            f"got {len(features)} features and {len(targets)} targets."
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets must contain only finite values.")
    return features, targets


# ---------------------------------------------------------------------------
# Plot dimension validation
# ---------------------------------------------------------------------------

def validate_plot_dimension(dimension, function_name):
    """Raise ValueError if dimension is not 2 or 3."""
    if dimension not in (2, 3):
        raise ValueError(
            f"{function_name} plots only support dimension=2 or dimension=3."
        )


# ---------------------------------------------------------------------------
# Class-name resolution
# ---------------------------------------------------------------------------

def get_class_names(labels, class_names):
    """Return labels, sorted unique label values, and resolved display names.

    Parameters
    ----------
    labels      : array-like, shape (N,)
    class_names : None | list | dict
        - None  → use str(label) for each unique label value.
        - list  → must follow np.unique(labels) order; one name per class.
        - dict  → keys are raw label values, values are display strings.

    Returns
    -------
    labels        : np.ndarray
    unique_labels : np.ndarray — sorted unique values
    class_names   : list[str]  — one display name per unique label
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if class_names is None:
        class_names = [str(label) for label in unique_labels]
    elif isinstance(class_names, dict):
        missing = [lbl for lbl in unique_labels if lbl not in class_names]
        if missing:
            raise ValueError(f"class_names is missing labels: {missing}")
        class_names = [class_names[lbl] for lbl in unique_labels]

    if len(class_names) != len(unique_labels):
        raise ValueError(
            f"class_names must have {len(unique_labels)} entries, "
            f"got {len(class_names)}"
        )
    return labels, unique_labels, class_names


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def get_plot_colors(n_classes, plt_colors=None):
    """Return one color per class without reusing colors.

    Parameters
    ----------
    n_classes  : int
    plt_colors : list | None
        When provided, must contain at least n_classes entries and is returned
        as-is. When None, colors are sampled from tab10 (≤10), tab20 (≤20),
        or hsv (>20).

    Returns
    -------
    list of matplotlib-compatible color values, length n_classes
    """
    if plt_colors is not None:
        if len(plt_colors) < n_classes:
            raise ValueError(
                f"Need at least {n_classes} colors for {n_classes} classes; "
                f"got {len(plt_colors)}."
            )
        return plt_colors

    colormap_name = (
        "tab10" if n_classes <= 10
        else "tab20" if n_classes <= 20
        else "hsv"
    )
    color_map = plt.colormaps[colormap_name].resampled(n_classes)
    return [color_map(i) for i in range(n_classes)]


# ---------------------------------------------------------------------------
# Scatter-plot primitive for labeled embeddings
# ---------------------------------------------------------------------------

def plot_labeled_embedding(
    embedding,
    labels,
    unique_labels,
    class_names,
    ax,
    plt_colors=None,
    size=15,
    alpha=0.7,
):
    """Scatter-plot a 2-D or 3-D embedding colored by class label.

    Parameters
    ----------
    embedding     : np.ndarray, shape (N, 2) or (N, 3)
    labels        : np.ndarray, shape (N,)
    unique_labels : np.ndarray — sorted unique class values
    class_names   : list[str]  — display name per class
    ax            : matplotlib Axes (2-D) or Axes3D (3-D)
    plt_colors    : list | None — passed to get_plot_colors
    size          : float — marker size
    alpha         : float — marker alpha
    """
    if embedding.shape[1] not in (2, 3):
        raise ValueError("plot_labeled_embedding requires 2 or 3 dimensions.")

    colors = get_plot_colors(len(unique_labels), plt_colors)
    for class_index, label_value in enumerate(unique_labels):
        mask = labels == label_value
        if not np.any(mask):
            continue
        if embedding.shape[1] == 2:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                color=colors[class_index],
                label=class_names[class_index],
                s=size,
                alpha=alpha,
            )
        else:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                color=colors[class_index],
                label=class_names[class_index],
                s=size,
                alpha=alpha,
            )


# ---------------------------------------------------------------------------
# Data-balancing utilities (originally from svm.py)
# ---------------------------------------------------------------------------

def balance_classes(X, y, samples, random_state=42):
    """Subsample each class to at most *samples* events.

    Parameters
    ----------
    X            : np.ndarray, shape (N, D)
    y            : np.ndarray, shape (N,)
    samples      : int — target count per class
    random_state : int

    Returns
    -------
    X_balanced : np.ndarray
    y_balanced : np.ndarray
    """
    np.random.seed(random_state)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = min(samples, int(np.min(class_counts)))

    X_parts, y_parts = [], []
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, min_count, replace=False)
        X_parts.append(X[chosen])
        y_parts.append(y[chosen])

    return np.vstack(X_parts), np.hstack(y_parts)


def sample_event_indices_by_label(labels, num_samples=10):
    """Return a dict mapping each unique label to a list of random row indices.

    Parameters
    ----------
    labels      : array-like, shape (N,)
    num_samples : int — max indices per label

    Returns
    -------
    dict[label, list[int]]
    """
    labels = np.asarray(labels)
    result = {}
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        result[label] = indices[:num_samples].tolist()
    return result


# ---------------------------------------------------------------------------
# Private-name aliases (for backward compatibility during migration)
# These allow clustering.py to keep using its leading-underscore names until
# the second migration step replaces each call-site with the public import.
# ---------------------------------------------------------------------------

_validate_features_and_labels = validate_features_and_labels
_validate_plot_dimension = validate_plot_dimension
_get_class_names = get_class_names
_get_plot_colors = get_plot_colors
_plot_labeled_embedding = plot_labeled_embedding
