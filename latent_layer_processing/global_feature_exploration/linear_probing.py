import os
import json
import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =========================== Utility Functions ===========================

def generate_log_train_sizes(min_size: int, max_size: int, num_points: int) -> np.ndarray:
    sizes = np.unique(np.round(np.logspace(np.log10(min_size), np.log10(max_size), num=num_points)).astype(int))
    sizes = np.unique(np.append(sizes, [min_size, max_size]))
    return np.sort(sizes)

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

def plot_confusion_matrix(y_true, y_pred, labels, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels,
                     square=True, cbar=True, linewidths=0.5, linecolor='black',
                     annot_kws={"size": 6})

    # Adjust fonts and labels
    plt.xlabel('Predicted Label', fontsize=6)
    plt.ylabel('True Label', fontsize=6)
    plt.title('Confusion Matrix', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n🧩 Confusion matrix saved to: {path}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None, normalize=False, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # show all ticks and label them
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')

    # rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate cells with counts or percentages
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6, fontweight='bold')

    ax.grid(False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
    plt.show()

# =========================== Main Linear Probing Function ===========================

@click.command()
@click.option('--beam', default='O16', help='Beam name')
@click.option('--num-points', default=512, help='Points per event')
@click.option('--num-classes', default=2, help='Number of output classes')
@click.option('--test-size', default=0.4, help='Test set fraction')
@click.option('--random-state', default=42, help='Random seed')
@click.option('--regularization', default=1.0, help='LogisticRegression C value')
@click.option('--min-train-size', default=50)
@click.option('--max-train-size', default=16000)
@click.option('--num-size-points', default=20)
@click.option('--cv-folds', default=3)
@click.argument('model_folder')
@click.argument('features')
@click.argument('labels')
def linear_probe_eval(beam, num_points, num_classes, test_size, random_state,
                      regularization, min_train_size, max_train_size, num_size_points,
                      cv_folds, model_folder, features, labels):

    output_dir = f"./{beam}_linear_probe"
    os.makedirs(output_dir, exist_ok=True)

    features = np.load(features)
    y = np.load(labels)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     features, y, test_size=test_size, random_state=random_state, stratify=y
    # )

    X_train = np.load('../global_features/X_train.npy')
    y_train = np.load('../global_features/y_train.npy')
    X_test = np.load('../global_features/X_test.npy')
    y_test = np.load('../global_features/y_test.npy')
    
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

    print("\nClassification Report on Test Set (Final Model):")

    final_clf = LogisticRegression(C=regularization, max_iter=1000, random_state=random_state)
    final_clf.fit(X_train_scaled, y_train)
    y_pred = final_clf.predict(X_test_scaled)
    
    target_names = [
        "0-, 1-, 2-track",  
        "3-track",       
        "4-, 5-track"     
    ]
    
    report = classification_report(y_test, y_pred, digits=4, target_names=target_names)
    print(report)

    plot_confusion_matrix(y_test, y_pred, target_names, output_dir)
    
if __name__ == '__main__':
    linear_probe_eval()
