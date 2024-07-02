import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from plotting import plot_learning_curve
from pointnet_model import pnet
from datetime import datetime
import click
import os

from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
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

# Load the saved model using Keras API
saved_model_path = 'models/2024-07-01-11:30:16/full_model' 
model = keras.models.load_model(saved_model_path, custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})

# Check if the model has been loaded correctly
print(model.summary())

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from plotting import plot_learning_curve
from tensorflow.keras.layers import GlobalAveragePooling1D

# Load the saved model using Keras API
saved_model_path = 'models/2024-07-01-11:30:16/full_model' 
model = load_model(saved_model_path, custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})

num_classes = 6  # Number of classes for the new task
num_epochs = 10
batch_size = 32

# Reconstruct the model without the last layer and add a new output layer with a unique name
# x = model.layers[-2].output
# print(x)
# print(model.layers[-1].output)
x = GlobalAveragePooling1D()(model.layers[-2].output)
new_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='reconstruction')(x)
model = tf.keras.models.Model(inputs=model.input, outputs=new_output)

# Load data and labels for the new task
train_data = np.load('ready/tpcnet_Mg22_expt_data_512.npy')
val_data = np.load('ready/tpcnet_Mg22_expt_labels_512.npy')

train_data, test_data, train_labels, test_labels = train_test_split(train_data, val_data, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size, drop_remainder=True)
val_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size, drop_remainder=True)

# Prepare for saving model checkpoints
timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
checkpoint_dir = f"Mg22_dw_models/{timestamp}/weights"
checkpoint_path = f"{checkpoint_dir}/cp-{{epoch:03d}}.ckpt"
os.makedirs(checkpoint_dir, exist_ok=True)

# Callbacks for training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=False, save_freq='epoch')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto', verbose=1, min_delta=0.001, min_lr=0.00001)

# Compile and fit model
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), metrics=["sparse_categorical_accuracy", "accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[checkpoint_callback, reduce_lr], verbose=1)

# Optional: Save learning curve plot
os.makedirs('Mg22_dw_plots/{}'.format(timestamp))
plot_learning_curve(history, 'Mg22_dw_plots/{}/learning_curve.png'.format(timestamp))

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

# Plot confusion matrix for the selected event
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Selected Event")
plt.savefig(f'Mg22_dw_plots/{timestamp}/confusion_matrix_event.png')
plt.show()

# Calculate mean accuracy and F1-score for the selected event
mean_accuracy = np.mean(y_true == y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Mean Accuracy for Selected Event: {mean_accuracy}")
print(f"F1-Score for Selected Event: {f1}")

# Save metrics for the selected event to a text file
with open(f'Mg22_dw_plots/{timestamp}/metrics_event.txt', 'w') as f:
    f.write(f"Mean Accuracy for Selected Event: {mean_accuracy}\n")
    f.write(f"F1-Score for Selected Event: {f1}\n")