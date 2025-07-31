import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf


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

def plot_accuracy(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
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
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

def plot_feature_importance(model, feature_names, output_dir, top_n=10):
    coef = model.coef_
    
    # extract numeric feature indices for sorting
    def get_feature_index(name):
        try:
            return int(name.split('_')[1])
        except (IndexError, ValueError):
            return float('inf')  # fallback for unexpected names
    
    for i, class_coef in enumerate(coef):
        # get top N absolute values
        top_indices = np.argsort(np.abs(class_coef))[::-1][:top_n]
        top_features = [(feature_names[idx], class_coef[idx]) for idx in top_indices]
        
        # sort top features by numeric feature index
        sorted_top = sorted(top_features, key=lambda x: get_feature_index(x[0]))
        sorted_names = [f[0] for f in sorted_top]
        sorted_vals = [f[1] for f in sorted_top]
        
        # plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(sorted_vals)), sorted_vals, tick_label=sorted_names)
        plt.title(f"Top Features for Class {i}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_class_{i}.png"))
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

    # Splits provided data into train and test data
def train_test_split(feature_data, label_data, samples):
    X = feature_data
    y = label_data

    # normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # shuffle and balance
    X, y = shuffle(X, y, random_state=42)
    X_balanced, y_balanced = balance_classes(X, y, samples)

    # train/test split
    X_train = X_balanced
    y_train = y_balanced

    # === uncomment if you want to remove global features used for training from the test set ===
    mask = ~np.any(np.all(X[:, None] == X_train, axis=2), axis=1)
    X_test = X[mask]
    y_test = y[mask]

    return X_train, y_train, X_test, y_test, X

    # Loads train, test data using provided ids
def load_train_test(feature_data, train_ids, train_labels, test_ids, test_labels):
    train_ids = np.array(train_ids, dtype=int)
    test_ids = np.array(test_ids, dtype=int)

    # slice features by ids
    X_train = feature_data[train_ids]
    y_train = np.array(train_labels)

    X_test = feature_data[test_ids]
    y_test = np.array(test_labels)

    X = np.concatenate([X_train, X_test], axis=0) 

    return X_train, y_train, X_test, y_test, X



  # === Neural network ===
def neural_network_classify(samples=300, num_classes=3, output_dir="neural_network_results"):
    os.makedirs(output_dir, exist_ok=True)

    X = np.load('../global_features/O16_Exp_extr.npy')
    y = np.load('../global_features/O16_Exp_labels.npy')
    
    X_train, y_train, X_test, y_test, X = train_test_split(X, y, samples)

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)    

    
    # define model 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
    
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
    
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-5
    )

    # train
    history = model.fit(
        X_train, y_train_cat,
        batch_size=8,
        epochs=100,
        verbose=1,
        callbacks=[lr_scheduler]
    )

    # plot
    plot_accuracy(history, output_dir)
    plot_loss(history, output_dir)

    print(f"Unbalanced Train Set: {(len(X_train)/len(X)):.2%}")
    print(f"Test Set: {(len(X_test)/len(X)):.2%}")

    
    # evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.2%}")

    y_test_pred_logits = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_logits, axis=1)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test F1 Score: {f1:.4f}")

    # predict on full dataset
    full_logits = model.predict(X)
    final_labels = np.argmax(full_logits, axis=1)  

    label_map = {
        0: "0-, 1-, 2-track", 1: "3-track", 2: "4-, 5-track"
    }

    # show label distribution
    print("\nPredicted label distribution:")
    summary_lines = []
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        line = f"{label_map.get(label, f'{label}-track'):<10}: {count}"
        print(line)
        summary_lines.append(line)

    # save
    np.save(os.path.join(output_dir, "O16_predicted_labels.npy"), final_labels)

    # sample indices by class
    if 'sample_event_indices_by_label' in globals():
        sampled = sample_event_indices_by_label(final_labels, num_samples=10)
        print("\nSampled Indices by Class:")
        for label, indices in sampled.items():
            print(f"{label_map.get(label, str(label)):<10}: {indices}")

    # confusion matrix
    class_names = ["0-, 1-, 2-track", "3-track", "4-, 5-track"]
    plot_confusion_matrix(y_test, y_test_pred, class_names, output_path=os.path.join(output_dir, "confusion_matrix.png"))

    return f1

if __name__ == '__main__':
    neural_network_classify()