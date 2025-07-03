import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle


def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None, normalize=False, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Show all ticks and label them
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts or percentages
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

def plot_accuracy(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    #plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()

def plot_loss(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    #plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

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
    min_count = samples

    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        selected_idx = np.random.choice(idx, min_count, replace=False)
        X_balanced.append(X[selected_idx])
        y_balanced.append(y[selected_idx])

    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)

    return X_balanced, y_balanced

def svm_classify2(output_dir="svm_results"):
    os.makedirs(output_dir, exist_ok=True)

    # Load experimental features and labels
    exp_features = np.load('../global_features/O16_experimental_features.npy')
    print(f"Loaded experimental features: {exp_features.shape}")

    label_data = np.load('../O16_Experimental_Labels.npy')
    indices = label_data[:, 0].astype(int)
    labels = label_data[:, 1].astype(int)
    print(f"Loaded labeled experimental events: {len(indices)}")
    
    # Extract labeled samples 
    X = exp_features[indices]
    y = labels

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X, y = shuffle(X, y, random_state=42)

    X_balanced, y_balanced = balance_classes(X, y, 10)

    print(f"Shape of X balanced {X_balanced.shape}")
    print(f"Shape of Y balanced {y_balanced.shape}")

    print(f"Using labeled experimental data: {X.shape}, {y.shape}")
    # === Train / Val / Test split ===
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    num_classes = 6 

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Define model 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-1))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_hinge',
        metrics=['accuracy']
    )

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=1e-5
    )

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=16,
        epochs=50,
        verbose=1,
        callbacks=[lr_scheduler]
    )

    # Plot
    plot_accuracy(history, output_dir)
    plot_loss(history, output_dir)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.2%}")

    y_test_pred_logits = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_logits, axis=1)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test F1 Score: {f1:.4f}")

    # Predict on full dataset
    full_logits = model.predict(X)
    final_labels = np.argmax(full_logits, axis=1)  

    label_map = {
        0: "0-track",
        1: "1-track",
        2: "2-track",
        3: "3-track",
        4: "4-track",
        5: "5-track",
    }

    # Show label distribution
    print("\nPredicted label distribution:")
    summary_lines = []
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        line = f"{label_map.get(label, f'{label}-track'):<10}: {count}"
        print(line)
        summary_lines.append(line)

    # Save
    np.save(os.path.join(output_dir, "O16_predicted_labels.npy"), final_labels)
    with open(os.path.join(output_dir, "O16_label_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    # Sample indices by class
    if 'sample_event_indices_by_label' in globals():
        sampled = sample_event_indices_by_label(final_labels, num_samples=10)
        print("\nSampled Indices by Class:")
        for label, indices in sampled.items():
            print(f"{label_map.get(label, str(label)):<10}: {indices}")

    # Confusion matrix
    class_names = ["0-track", "1-track", "2-track", "3-track", "4-track", "5-track"]
    plot_confusion_matrix(y_test, y_test_pred, class_names, output_path=os.path.join(output_dir, "confusion_matrix.png"))


def svm_classify(samples, output_dir="svm_results"):
    os.makedirs(output_dir, exist_ok=True)

    # Load experimental features and labels
    exp_features = np.load('../global_features/O16_experimental_features.npy')
    print(f"Loaded experimental features: {exp_features.shape}")

    label_data = np.load('../O16_Experimental_Labels.npy')
    indices = label_data[:, 0].astype(int)
    labels = label_data[:, 1].astype(int)
    print(f"Loaded labeled experimental events: {len(indices)}")
    
    # Extract labeled samples 
    X = exp_features[indices]
    y = labels

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X, y = shuffle(X, y, random_state=42)

    X_balanced, y_balanced = balance_classes(X, y, samples)

    print(f"Shape of X balanced {X_balanced.shape}")
    print(f"Shape of Y balanced {y_balanced.shape}")

    print(f"Using labeled experimental data: {X.shape}, {y.shape}")
    # === Train / Val / Test split ===
    X_train = X_balanced
    y_train = y_balanced
    X_test = X
    y_test = y
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    num_classes = 6 

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)    
    
    # Define model 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_hinge',
        metrics=['accuracy']
    )

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=1e-5
    )

    # Train
    history = model.fit(
        X_train, y_train_cat,
        batch_size=16,
        epochs=200,
        verbose=1,
        callbacks=[lr_scheduler]
    )

    # Plot
    plot_accuracy(history, output_dir)
    plot_loss(history, output_dir)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.2%}")

    y_test_pred_logits = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_logits, axis=1)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test F1 Score: {f1:.4f}")

    # Predict on full dataset
    full_logits = model.predict(X)
    final_labels = np.argmax(full_logits, axis=1)  

    label_map = {
        0: "0-track",
        1: "1-track",
        2: "2-track",
        3: "3-track",
        4: "4-track",
        5: "5-track",
    }

    # Show label distribution
    print("\nPredicted label distribution:")
    summary_lines = []
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        line = f"{label_map.get(label, f'{label}-track'):<10}: {count}"
        print(line)
        summary_lines.append(line)

    # Save
    np.save(os.path.join(output_dir, "O16_predicted_labels.npy"), final_labels)
    with open(os.path.join(output_dir, "O16_label_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    # Sample indices by class
    if 'sample_event_indices_by_label' in globals():
        sampled = sample_event_indices_by_label(final_labels, num_samples=10)
        print("\nSampled Indices by Class:")
        for label, indices in sampled.items():
            print(f"{label_map.get(label, str(label)):<10}: {indices}")

    # Confusion matrix
    class_names = ["0-track", "1-track", "2-track", "3-track", "4-track", "5-track"]
    plot_confusion_matrix(y_test, y_test_pred, class_names, output_path=os.path.join(output_dir, "confusion_matrix.png"))

    return test_acc

if __name__ == '__main__':
    svm_classify()
