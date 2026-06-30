import click
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def validate_features_and_labels(features, labels):
    """Validate that every feature row has exactly one class label."""
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
    """Validate that every feature row has exactly one regression target."""
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


def save_classification_outputs(y_test, y_pred, classes, class_names, results_folder):
    """Save the classification report and confusion matrix for the linear probe."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("Actual Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.title("Linear Probe Confusion Matrix", fontsize=14, fontweight="bold", pad=15)

    plt.savefig(f"{results_folder}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report_dict).transpose().round(4)
    df.to_csv(f"{results_folder}/classification_report.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title("Classification Report", fontsize=14, fontweight="bold", pad=15)
    plt.savefig(f"{results_folder}/classification_report.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_regression_outputs(y_test, y_pred, results_folder):
    """Save predicted-vs-true plot and regression metrics table."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, s=20)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Ideal")
    plt.xlabel("True Value", fontsize=11)
    plt.ylabel("Predicted Value", fontsize=11)
    plt.title("Linear Probe Predicted vs True", fontsize=14, fontweight="bold", pad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_folder}/predicted_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close()


def get_class_names(classes, class_names=None):
    """Use provided class names, or fall back to simple numeric labels."""
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


def compute_regression_metrics(y_true, y_pred):
    """Return standard regression metrics for a single target."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def scale_probe_features(X_train, X_test):
    """Fit StandardScaler on the training split and transform both splits."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def run_classification_probe(
    X_train,
    X_test,
    y_train,
    y_test,
    classifier,
    regularization,
    base_seed,
    class_names_option,
    results_folder,
):
    unique_classes = np.unique(y_train)
    class_names = get_class_names(unique_classes, class_names_option)

    print(f"\nTraining linear probe ({classifier})...")
    if classifier == "logistic-regression":
        model = LogisticRegression(
            C=regularization,
            random_state=base_seed,
            max_iter=5000,
        )
    else:
        model = LinearSVC(
            C=regularization,
            random_state=base_seed,
            max_iter=10000,
            dual="auto",
        )

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Linear probe - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    save_classification_outputs(y_test, y_test_pred, unique_classes, class_names, results_folder)

    results = {
        "experiment_config": {
            "task": "classification",
            "classifier": classifier,
            "regularization": regularization,
            "class_names": class_names,
        },
        "metrics": {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
        },
    }

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=unique_classes,
            target_names=class_names,
            zero_division=0,
        )
    )

    return results


def run_regression_probe(X_train, X_test, y_train, y_test, regularization, results_folder):
    print("\nTraining linear regression probe (Ridge)...")
    model = Ridge(alpha=regularization)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    print(
        "Linear regression probe - "
        f"Train R2: {train_metrics['r2']:.4f}, Test R2: {test_metrics['r2']:.4f}, "
        f"Test RMSE: {test_metrics['rmse']:.4f}"
    )

    save_regression_outputs(y_test, y_test_pred, results_folder)

    results = {
        "experiment_config": {
            "task": "regression",
            "model": "ridge",
            "regularization": regularization,
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
    }

    return results


@click.command()
@click.option(
    "--name",
    default="O16",
    type=click.STRING,
    help="The name/profile identifier for the run (e.g. O16, Mg22, C16)",
)
@click.option("--test-size", default=0.2, type=click.FLOAT, help="Fraction of data held out for testing")
@click.option("--seed", default=None, type=click.INT, help="Random seed for reproducibility")
@click.option(
    "--task",
    type=click.Choice(["classification", "regression"]),
    default="classification",
    help="Probe task type: classification or regression",
)
@click.option(
    "--regularization",
    default=1.0,
    type=click.FLOAT,
    help="Classification: classifier C (larger = weaker regularization). Regression: Ridge alpha.",
)
@click.option(
    "--classifier",
    type=click.Choice(["logistic-regression", "linear-svm"]),
    default="logistic-regression",
    help="Linear classifier for classification tasks",
)
@click.option(
    "--class-name",
    "class_names",
    multiple=True,
    help="Class display name. Repeat once per sorted unique label.",
)
@click.argument("features-file", type=click.Path(exists=True))
@click.argument("labels-file", type=click.Path(exists=True))
def linear_probe_evaluation(
    name,
    test_size,
    seed,
    task,
    regularization,
    classifier,
    class_names,
    features_file,
    labels_file,
):
    """Evaluate frozen embeddings with a linear probe for classification or regression."""

    print("Loading features and labels...")
    global_features = np.load(features_file)
    target_values = np.load(labels_file)

    print(f"Features shape: {global_features.shape}")
    print(f"Targets shape: {np.asarray(target_values).shape}")

    base_seed = seed if seed is not None else np.random.randint(0, 100000)
    print(f"Using random seed baseline: {base_seed}")

    master_results_dir = "./linear_probe_results"
    results_suffix = "linear_regression" if task == "regression" else "linear_probe"
    results_folder = os.path.join(master_results_dir, f"{name}_{results_suffix}")
    os.makedirs(results_folder, exist_ok=True)
    print(f"Target results directory established: {results_folder}")

    if task == "classification":
        global_features, target_values = validate_features_and_labels(global_features, target_values)
        split_kwargs = {"stratify": target_values}
    else:
        global_features, target_values = validate_features_and_targets(global_features, target_values)
        split_kwargs = {}

    X_train, X_test, y_train, y_test = train_test_split(
        global_features,
        target_values,
        test_size=test_size,
        random_state=base_seed,
        **split_kwargs,
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    print("Standardizing features (train-split fit)...")
    X_train_scaled, X_test_scaled, _ = scale_probe_features(X_train, X_test)

    if task == "classification":
        task_results = run_classification_probe(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            classifier,
            regularization,
            base_seed,
            class_names,
            results_folder,
        )
    else:
        task_results = run_regression_probe(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            regularization,
            results_folder,
        )

    results = {
        "dataset_info": {
            "features_source": features_file,
            "labels_source": labels_file,
            "total_samples": len(global_features),
            "feature_dim": global_features.shape[1],
        },
        "experiment_config": {
            "test_size": test_size,
            "seed": seed,
            "effective_seed": int(base_seed),
            **task_results["experiment_config"],
        },
        "metrics": task_results["metrics"],
    }

    with open(f"{results_folder}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nAll results saved to: {results_folder}")


if __name__ == "__main__":
    linear_probe_evaluation()
