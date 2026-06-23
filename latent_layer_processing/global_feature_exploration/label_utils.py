"""Helpers for verified reference labels and partially labeled datasets."""

import numpy as np

# Integer rows with this value are treated as unlabeled by default.
DEFAULT_UNLABELED_VALUES = (-1,)


def normalize_unlabeled_values(unlabeled_values):
    if unlabeled_values is None:
        return DEFAULT_UNLABELED_VALUES
    if np.isscalar(unlabeled_values):
        return (unlabeled_values,)
    return tuple(unlabeled_values)


def labeled_mask(labels, unlabeled_values=None):
    """Return True for rows with a verified class label."""
    labels = np.asarray(labels).ravel()
    mask = np.ones(len(labels), dtype=bool)

    for value in normalize_unlabeled_values(unlabeled_values):
        mask &= labels != value
        if labels.dtype.kind in {"U", "S", "O"}:
            mask &= labels != str(value)

    if np.issubdtype(labels.dtype, np.floating):
        mask &= ~np.isnan(labels)

    return mask


def split_labeled(features, labels, unlabeled_values=None):
    """Return labeled rows only, preserving row alignment before filtering."""
    features = np.asarray(features)
    labels = np.asarray(labels).ravel()
    if len(features) != len(labels):
        raise ValueError(
            "features and labels must contain the same number of rows; "
            f"got {len(features)} features and {len(labels)} labels."
        )

    mask = labeled_mask(labels, unlabeled_values)
    return features[mask], labels[mask], mask


def summarize_label_coverage(labels, unlabeled_values=None):
    """Report how many events are labeled vs unlabeled."""
    labels = np.asarray(labels).ravel()
    mask = labeled_mask(labels, unlabeled_values)
    n_total = len(labels)
    n_labeled = int(mask.sum())
    n_unlabeled = n_total - n_labeled
    return {
        "n_total": n_total,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "labeled_fraction": n_labeled / n_total if n_total else 0.0,
        "labeled_mask": mask,
    }


def require_labeled_rows(labels, unlabeled_values=None, context="This analysis"):
    """Raise if no labeled rows remain; warn when some rows are ignored."""
    coverage = summarize_label_coverage(labels, unlabeled_values)
    if coverage["n_labeled"] == 0:
        raise ValueError(
            f"{context} requires at least one verified labeled event. "
            f"All {coverage['n_total']} rows appear unlabeled. "
            "Use verified reference labels, or run label-free analyses such as "
            "PCA variance only."
        )
    if coverage["n_unlabeled"] > 0:
        print(
            f"{context}: using {coverage['n_labeled']} labeled events and "
            f"ignoring {coverage['n_unlabeled']} unlabeled rows."
        )
    return coverage


def print_label_coverage(coverage):
    print(
        f"Label coverage: {coverage['n_labeled']}/{coverage['n_total']} labeled "
        f"({coverage['labeled_fraction']:.1%}), "
        f"{coverage['n_unlabeled']} unlabeled"
    )
