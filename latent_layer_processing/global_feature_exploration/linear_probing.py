import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


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


def save_linear_probe_outputs(y_test, y_pred, classes, class_names, results_folder):
    """
    Save the classification report and confusion matrix for the linear probe.
    """
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Actual Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.title('Linear Probe Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    
    plt.savefig(f'{results_folder}/confusion_matrix.png', dpi=300, bbox_inches='tight')
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

    df.to_csv(f'{results_folder}/classification_report.csv')


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


@click.command()
@click.option('--name', default='O16', type=click.STRING, help='The name/profile identifier for the run (e.g. O16, Mg22, C16)')
@click.option('--test-size', default=0.2, type=click.FLOAT, help='Fraction of data to use for testing')
@click.option('--seed', default=None, type=click.INT, help='Random seed for reproducibility')
@click.option('--regularization', default=1.0, type=click.FLOAT, help='Classifier C value; larger values mean weaker regularization')
@click.option('--classifier', type=click.Choice(['logistic-regression', 'linear-svm']), default='logistic-regression', help='Linear classifier to train on frozen embeddings')
@click.option('--class-name', 'class_names', multiple=True, help='Class display name. Repeat once per sorted unique label.')
@click.argument('features-file', type=click.Path(exists=True))
@click.argument('labels-file', type=click.Path(exists=True))
def linear_probe_evaluation(name, test_size, seed, regularization, classifier, class_names, features_file, labels_file):
    """
    Perform linear probe evaluation using pre-extracted NumPy feature embeddings
    and corresponding target labels.
    """

    print("Loading features and labels...")

    global_features = np.load(features_file) # Expected shape: (N, D)
    combined_track_labels = np.load(labels_file) # Expected shape: (N,)
    global_features, combined_track_labels = validate_features_and_labels(
        global_features,
        combined_track_labels,
    )
    unique_classes = np.unique(combined_track_labels)

    print(f"Features shape: {global_features.shape}")
    print(f"Labels shape: {combined_track_labels.shape}")

    base_seed = seed if seed is not None else np.random.randint(0, 100000)
    print(f"Using random seed baseline: {base_seed}")
    
    master_results_dir = "./linear_probe_results"
    results_folder = os.path.join(master_results_dir, f"{name}_linear_probe")
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"Target results directory established: {results_folder}")
    
    # Split data into train and test sets (fixed test set for consistent evaluation)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        global_features, combined_track_labels, 
        test_size=test_size, random_state=base_seed, 
        stratify=combined_track_labels
    )
    
    print(f"Full training set size: {X_train_full.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Standardize features (fit on full training set)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining linear probe ({classifier})...")
    if classifier == 'logistic-regression':
        linear_probe = LogisticRegression(
            C=regularization,
            random_state=base_seed,
            max_iter=5000
        )
    else:
        linear_probe = LinearSVC(
            C=regularization,
            random_state=base_seed,
            max_iter=10000,
            dual='auto'
        )
    linear_probe.fit(X_train_full_scaled, y_train_full)
    
    y_train_pred = linear_probe.predict(X_train_full_scaled)
    y_test_pred = linear_probe.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train_full, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Linear probe - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    class_names = get_class_names(unique_classes, class_names)
    save_linear_probe_outputs(y_test, y_test_pred, unique_classes, class_names, results_folder)

    results = {
        'dataset_info': {
            'features_source': features_file,
            'labels_source': labels_file,
            'total_samples': len(global_features),
            'feature_dim': global_features.shape[1],
        },
        'experiment_config': {
            'test_size': test_size,
            'regularization': regularization,
            'classifier': classifier,
            'seed': seed,
            'effective_seed': int(base_seed),
            'class_names': class_names,
        },
        'metrics': {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
        }
    }

    with open(f'{results_folder}/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll results saved to: {results_folder}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        labels=unique_classes,
        target_names=class_names,
        zero_division=0,
    ))


if __name__ == '__main__':
    linear_probe_evaluation()
