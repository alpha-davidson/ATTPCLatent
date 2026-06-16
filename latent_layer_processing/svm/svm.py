import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import train_test_split
import click

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



def sample_event_indices_by_label(labels, num_samples=10):
    sampled_indices = {}
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        sampled_indices[label] = indices[:num_samples].tolist()
    return sampled_indices

def balance_classes(X, y, samples, random_state=42):
    np.random.seed(random_state)
    X_balanced = []
    y_balanced = []

    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_available = np.min(class_counts)
    min_count = min(samples, min_available)

    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        selected_idx = np.random.choice(idx, min_count, replace=False)
        X_balanced.append(X[selected_idx])
        y_balanced.append(y[selected_idx])

    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)

    return X_balanced, y_balanced





@click.command()
@click.argument('features', type=click.Path(exists=True))
@click.argument('labels', type=click.Path(exists=True))
@click.option('--samples', type=click.INT, default=10, help='Number of samples per class for balancing')
@click.option('--output-dir', default="svm_results", type=click.STRING, help='Output directory for plots and predictions')
def svm_classify(features, labels, samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X = np.load(features)
    y = np.load(labels)

    

    X_balanced, y_balanced = balance_classes(X, y, samples)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_balanced, 
        y_balanced, 
        test_size=0.2, 
        stratify=y_balanced, 
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_all_scaled = scaler.transform(X)


    print(f"Train Subspace: {X_train.shape} | Test Subspace: {X_test.shape}")

    model = svm.LinearSVC(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    # predict
    y_test_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    acc = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    # predict full dataset
    final_labels = model.predict(X_all_scaled)

    unique_labels = np.unique(y)
    label_map = {int(lbl): f"Class {int(lbl)}-track" for lbl in unique_labels}
    
    print("\nPredicted label distribution:")
    summary_lines = []
    unique_preds, counts = np.unique(final_labels, return_counts=True)
    for label, count in zip(unique_preds, counts):
        line = f"{label_map.get(label, f'{label}-track'):<10}: {count}"
        print(line)
        summary_lines.append(line)

    # save results
    np.save(os.path.join(output_dir, "predicted_labels.npy"), final_labels)

    # confusion matrix
    class_names = [label_map[l] for l in sorted(unique_labels)]
    plot_confusion_matrix(y_test, y_test_pred, class_names, output_path=os.path.join(output_dir, "confusion_matrix.png"))
    
    return f1
    
if __name__ == '__main__':
    svm_classify()