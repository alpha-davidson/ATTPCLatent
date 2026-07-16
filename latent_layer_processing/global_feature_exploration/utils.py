"""Shared validation, labeling, and plotting helpers reused by clustering.py
and linear_probing.py."""

import matplotlib.pyplot as plt
import numpy as np


def validate_features_and_labels(features, labels):
    """Validate that every feature row has exactly one label."""
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


def validate_plot_dimension(dimension, function_name):
    if dimension not in (2, 3):
        raise ValueError(f"{function_name} plots only support dimension=2 or dimension=3.")


def get_class_names(labels, class_names):
    """Return labels, sorted label values, and display names.

    labels should contain one class label for each row in the feature matrix.
    If class_names is a list, it must follow the order of np.unique(labels).
    If class_names is a dict, keys should be raw label values and values are
    the display names used in plot legends.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [str(label) for label in unique_labels]
    elif isinstance(class_names, dict):
        missing_labels = [label for label in unique_labels if label not in class_names]
        if missing_labels:
            raise ValueError(f"class_names is missing labels: {missing_labels}")
        class_names = [class_names[label] for label in unique_labels]
    if len(class_names) != len(unique_labels):
        raise ValueError(
            f"class_names must have {len(unique_labels)} entries, got {len(class_names)}"
        )
    return labels, unique_labels, class_names


def format_class_names(classes, class_names=None):
    """Use provided class names, or fall back to simple numeric labels.

    Unlike get_class_names, this does not compute unique labels itself: it
    formats display names for a caller-provided sequence of class values,
    rendering whole-number floats without a trailing ".0".
    """
    if class_names:
        if len(class_names) != len(classes):
            raise ValueError(
                f"Expected {len(classes)} class names, got {len(class_names)}. "
                "Provide one --class-name value per unique label."
            )
        return list(class_names)

    numeric_class_names = []
    for cls in classes:
        try:
            value = float(cls)
        except (TypeError, ValueError):
            numeric_class_names.append(str(cls))
        else:
            numeric_class_names.append(str(int(value)) if value.is_integer() else str(cls))
    return numeric_class_names


def get_plot_colors(n_classes, plt_colors=None):
    """Return one color per class without reusing colors.

    If plt_colors is provided, it must include at least one color per class.
    Otherwise, colors are generated from tab10 for up to 10 classes, tab20 for
    up to 20 classes, and hsv for larger class counts.
    """
    if plt_colors is not None:
        if len(plt_colors) < n_classes:
            raise ValueError(
                f"Need at least {n_classes} colors for {n_classes} classes; "
                f"got {len(plt_colors)}."
            )
        return plt_colors

    colormap_name = "tab10" if n_classes <= 10 else "tab20" if n_classes <= 20 else "hsv"
    color_map = plt.colormaps[colormap_name].resampled(n_classes)
    return [color_map(i) for i in range(n_classes)]


def plot_labeled_embedding(embedding, labels, unique_labels, class_names, ax,
                            plt_colors=None, size=15, alpha=0.7):
    if embedding.shape[1] not in (2, 3):
        raise ValueError("Labeled embedding plots require 2 or 3 dimensions.")

    colors = get_plot_colors(len(unique_labels), plt_colors)
    for class_index, label_value in enumerate(unique_labels):
        mask = (labels == label_value)
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
