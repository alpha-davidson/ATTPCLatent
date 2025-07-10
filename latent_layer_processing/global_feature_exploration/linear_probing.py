import os
import json
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ATTPCLatent.pointnet import create_pointnet_model


# =========================== Utility Functions ===========================

def generate_log_train_sizes(min_size: int, max_size: int, num_points: int) -> np.ndarray:
    sizes = np.unique(np.round(np.logspace(np.log10(min_size), np.log10(max_size), num=num_points)).astype(int))
    sizes = np.unique(np.append(sizes, [min_size, max_size]))
    return np.sort(sizes)

def load_pointnet_model(path: str) -> keras.Model:
    print("Loading PointNet model...")
    model = tf.keras.models.load_model(path)
    print("Model loaded.")
    return model

def extract_latent_features(model: keras.Model, dataset: np.ndarray, batch_size: int = 32) -> np.ndarray:
    try:
        extractor = keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
    except ValueError:
        extractor = keras.Model(inputs=model.input, outputs=model.layers[-3].output)
        print(f"Using fallback layer '{model.layers[-3].name}' for feature extraction.")
    
    ds = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)
    return extractor.predict(ds, verbose=1)

def load_data(file_2track: str, file_3track: str, num_features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    data_2 = np.load(f'{file_2track}.npy')[:, :, :num_features]
    data_3 = np.load(f'{file_3track}.npy')[:, :, :num_features]
    X = np.concatenate([data_2, data_3], axis=0)
    y = np.concatenate([np.zeros(len(data_2)), np.ones(len(data_3))])
    return X, y.astype(int)

def stratified_sample(X, y, size: int, num_classes: int, seed: int = 0):
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_size = size // num_classes + (size % num_classes if cls == np.unique(y)[-1] else 0)
        cls_sample = np.random.RandomState(seed).choice(cls_idx, min(cls_size, len(cls_idx)), replace=False)
        indices.extend(cls_sample)
    return X[indices], y[indices]

# =========================== Visualization ===========================

def plot_learning_curve(results: dict, out_dir: str):
    sizes = results['train_sizes']
    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, results['train_accuracies_mean'], yerr=results['train_accuracies_std'], label='Train', fmt='-o')
    plt.errorbar(sizes, results['test_accuracies_mean'], yerr=results['test_accuracies_std'], label='Test', fmt='-o')
    plt.xscale('log')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curve.png'))
    plt.close()

def print_results_table(results, output_dir):
    headers = ['Train Size', 'Train Accuracy (mean ± std)', 'Test Accuracy (mean ± std)']
    col_widths = [12, 30, 30]

    # Header
    print("\n📊 Linear Probing Accuracy Summary:\n")
    header_row = "| " + " | ".join(h.center(w) for h, w in zip(headers, col_widths)) + " |"
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    print(separator)
    print(header_row)
    print(separator)

    # Rows
    for i, size in enumerate(results['train_sizes']):
        row = [
            str(size).rjust(col_widths[0]),
            f"{results['train_accuracies_mean'][i]:.4f} ± {results['train_accuracies_std'][i]:.4f}".center(col_widths[1]),
            f"{results['test_accuracies_mean'][i]:.4f} ± {results['test_accuracies_std'][i]:.4f}".center(col_widths[2]),
        ]
        print("| " + " | ".join(row) + " |")
    print(separator)
    print(f"\n🖼️  Learning curve saved to: {os.path.join(output_dir, 'learning_curve.png')}")

# =========================== Main Linear Probing Function ===========================

@click.command()
@click.option('--beam', default='O16', help='Beam name')
@click.option('--num-points', default=512, help='Points per event')
@click.option('--num-classes', default=2, help='Number of output classes')
@click.option('--test-size', default=0.2, help='Test set fraction')
@click.option('--random-state', default=42, help='Random seed')
@click.option('--regularization', default=1.0, help='LogisticRegression C value')
@click.option('--min-train-size', default=50)
@click.option('--max-train-size', default=16000)
@click.option('--num-size-points', default=20)
@click.option('--cv-folds', default=3)
@click.argument('model_folder')
@click.argument('data_file_stem_2track')
@click.argument('data_file_stem_3track')
def linear_probe_eval(beam, num_points, num_classes, test_size, random_state,
                      regularization, min_train_size, max_train_size, num_size_points,
                      cv_folds, model_folder, data_file_stem_2track, data_file_stem_3track):

    output_dir = f"./{beam}_linear_probe"
    os.makedirs(output_dir, exist_ok=True)

    X_raw, y = load_data(data_file_stem_2track, data_file_stem_3track)
    model = load_pointnet_model(model_folder)
    features = extract_latent_features(model, X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_sizes = generate_log_train_sizes(min_train_size, min(max_train_size, len(X_train)), num_size_points)

    results = {
        'train_sizes': train_sizes.tolist(),
        'train_accuracies_mean': [],
        'train_accuracies_std': [],
        'test_accuracies_mean': [],
        'test_accuracies_std': []
    }

    for size in train_sizes:
        train_accs, test_accs = [], []
        for fold in range(cv_folds):
            X_sub, y_sub = stratified_sample(X_train_scaled, y_train, size, num_classes, seed=random_state + fold)
            clf = LogisticRegression(C=regularization, max_iter=1000, random_state=random_state + fold)
            clf.fit(X_sub, y_sub)

            train_accs.append(accuracy_score(y_sub, clf.predict(X_sub)))
            test_accs.append(accuracy_score(y_test, clf.predict(X_test_scaled)))

        results['train_accuracies_mean'].append(np.mean(train_accs))
        results['train_accuracies_std'].append(np.std(train_accs))
        results['test_accuracies_mean'].append(np.mean(test_accs))
        results['test_accuracies_std'].append(np.std(test_accs))

    plot_learning_curve(results, output_dir)
    print_results_table(results, output_dir)

if __name__ == '__main__':
    linear_probe_eval()
