import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from datetime import datetime
import click
import os
import sys
import random

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from plotting import plot_learning_curve
from pointnet_model import pnet
from plotting import plot_events, plot_histogram, plot_zero_one_bins

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    def get_config(self):
        return {'l2reg': self.l2reg, 'num_features': self.num_features}

def plot_single_event(test_data, true_label, predicted_label, idx, labels, save_path):
    """
    Plot a single mislabeled event in 3D, annotating its true and predicted labels.

    :param test_data: The test dataset (features).
    :param true_label: True label for the event.
    :param predicted_label: Predicted label for the event.
    :param idx: Index of the event in the test data.
    :param labels: List of class names or numbers for the labels.
    :param save_path: Path to save the 3D plot.
    """
    mislabeled_event = test_data[idx, :, :]  # Extract the mislabeled event
    mislabeled_x = mislabeled_event[:, 0]
    mislabeled_y = mislabeled_event[:, 1]
    mislabeled_z = mislabeled_event[:, 2]

    # Convert true_label and predicted_label to integers
    true_label = int(true_label)
    predicted_label = int(predicted_label)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mislabeled_x, mislabeled_y, mislabeled_z, s=5, c='red', label='Mislabeled Event')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Add labels for the event
    ax.set_title(f"True Label: {labels[true_label]}, Predicted Label: {labels[predicted_label]}")
    plt.legend()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close(fig)

def main():
    # Load the saved model using Keras API
    saved_model_path = '../../../O16_models/2024-10-01-11:57:25/full_model' 
    model = keras.models.load_model(saved_model_path, custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
    
    # Check if the model has been loaded correctly
    print(model.summary())
    
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from sklearn.metrics import classification_report
    
    # Load the saved model using Keras API
    saved_model_path = '../../../O16_models/2024-10-01-11:57:25/full_model' 
    model = load_model(saved_model_path, custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
    
    num_classes = 6  # Number of classes for the new task
    num_epochs = 10
    batch_size = 32

    # Reconstruct the model without the last layer and add a new output layer with a unique name
    # x = model.layers[-2].output
    # print(x)
    # print(model.layers[-1].output)
    x = GlobalAveragePooling1D()(model.layers[-2].output)
    #new_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='reconstruction')(x)
    #model = tf.keras.models.Model(inputs=model.input, outputs=new_output)
    
    # Load data and labels for the new task
    #train_data = np.load('ready/tpcnet_Mg22_expt_data_512.npy')
    #val_data = np.load('ready/tpcnet_Mg22_expt_labels_512.npy')
    
    
    new_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='reconstruction')(x)
    model = tf.keras.models.Model(inputs=model.input, outputs=new_output)
    
    train_data = np.load('../data_processing/data_splits/O16_size512_train_features.npy')
    train_labels = np.load('../data_processing/data_splits/O16_size512_train_labels.npy')
    val_data = np.load('../data_processing/data_splits/O16_size512_val_features.npy')
    val_labels = np.load('../data_processing/data_splits/O16_size512_val_labels.npy')
    test_data = np.load('../data_processing/data_splits/O16_size512_test_features.npy')
    test_labels = np.load('../data_processing/data_splits/O16_size512_test_labels.npy')
        
    train_data = train_data[:, :, :3]
    val_data = val_data[:, :, :3]
    test_data = test_data[:, :, :3]
        
    #train_data, test_data, train_labels, test_labels = train_test_split(train_data, val_data, test_size=0.2, random_state=42)
        
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size, drop_remainder=True)
    val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size, drop_remainder=True)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size, drop_remainder=True)
        
    # Prepare for saving model checkpoints
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    checkpoint_dir = f"pretrain_models/{timestamp}/weights"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/cp-{{epoch:03d}}.ckpt"
        
    # Callbacks for training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=False, save_freq='epoch')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto', verbose=1, min_delta=0.001, min_lr=0.00001)
        
    # Compile and fit model
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), metrics=["sparse_categorical_accuracy", "accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[checkpoint_callback, reduce_lr], verbose=1)
        
    # Optional: Save learning curve plot
    directory_path = f"../evaluation_and_plotting/{timestamp}"
    os.makedirs(directory_path, exist_ok=True)
    plot_learning_curve(history, f'../evaluation_and_plotting/{timestamp}/learning_curve.png')
        
    # Select specific event based on the lowest validation loss
    best_epoch = np.argmin(history.history['val_loss'])
    print(f"Best epoch based on validation loss: {best_epoch}")
        
    # Make predictions for the selected event (best_epoch)
    model.load_weights(checkpoint_path.format(epoch=best_epoch + 1))  # SLURM indexing starts from 1
    y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = test_labels

        
    # Compute confusion matrix for the selected event
    conf_matrix = confusion_matrix(y_true, y_pred)
        
    # Print confusion matrix for the selected event
    print("Confusion Matrix for Selected Event:")
    print(conf_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0-Track", "1-Track", "2-Track", "3-Track", "4-5 Tracks"], 
                yticklabels=["0-Track", "1-Track", "2-Track", "3-Track", "4-5 Tracks"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(f'../evaluation_and_plotting/{timestamp}/confusion_matrix_event.png')
    plt.show()
        
    # Calculate mean accuracy and F1-score for the selected event
    mean_accuracy = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
        
    print(f"Mean Accuracy for Selected Event: {mean_accuracy}")
    print(f"F1-Score for Selected Event: {f1}")
    print(classification_report(y_true, y_pred))
    
    # Save metrics for the selected event to a text file
    with open(f'../evaluation_and_plotting/{timestamp}/metrics_event.txt', 'w') as f:
        f.write(f"Mean Accuracy for Selected Event: {mean_accuracy}\n")
        f.write(f"F1-Score for Selected Event: {f1}\n")

    mislabeled_indices = np.where((y_pred != y_true) & (y_true == 3))[0]
    correct_indices = np.where((y_pred == y_true) & (y_true == 3))[0]

    # Randomly select up to 10 events for mislabeled and correct
    mislabeled_indices = random.sample(list(mislabeled_indices), min(10, len(mislabeled_indices)))
    correct_indices = random.sample(list(correct_indices), min(10, len(correct_indices)))

    labels = [f'Class {i}' for i in range(num_classes)]  # Example class labels

    # Create directories for mislabeled and correct events
    mislabeled = f'../evaluation_and_plotting/{timestamp}/mislabeled'
    correct = f'../evaluation_and_plotting/{timestamp}/correct'
    os.makedirs(mislabeled, exist_ok=True)
    os.makedirs(correct, exist_ok=True)

    # Save mislabeled events
    for i, idx in enumerate(mislabeled_indices):
        true_label = y_true[idx]
        predicted_label = y_pred[idx]
        save_path = os.path.join(
            mislabeled, f"mislabeled_event_{i}_true_{int(true_label)}_pred_{int(predicted_label)}.png")
        plot_single_event(test_data, true_label, predicted_label, idx, labels, save_path)
        print(f"Saved mislabeled event {i} with true label {int(true_label)} and predicted label {int(predicted_label)} to {save_path}")

    # Save correctly labeled events
    for i, idx in enumerate(correct_indices):
        true_label = y_true[idx]
        predicted_label = y_pred[idx]
        save_path = os.path.join(
            correct, f"correct_event_{i}_true_{int(true_label)}_pred_{int(predicted_label)}.png")
        plot_single_event(test_data, true_label, predicted_label, idx, labels, save_path)
        print(f"Saved correctly labeled event {i} with true label {int(true_label)} and predicted label {int(predicted_label)} to {save_path}")
    

if __name__ == '__main__':
    main()
