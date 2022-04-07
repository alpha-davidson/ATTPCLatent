import os
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cnn_functions as cnn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
layers = tf.keras.layers


#builds a VGG16 CNN model with some extra layers
def build_pretrained_vgg_model(input_shape, num_classes):
    vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    net = layers.Flatten()(vgg_model.output)
    net = layers.Dense(512, activation=tf.nn.relu)(net)   #two fully-connected layers that take in the flattened features
    net = layers.Dense(512, activation=tf.nn.relu)(net)
    output = layers.Dense(num_classes, activation=tf.nn.softmax)(net)   #groups images into one of three classes
    model = tf.keras.Model(inputs=vgg_model.input, outputs=output)   #model whose input is VGG16's and output is that of the layer above
    for layer in model.layers[:-4]:    #specifying that the last four layers (the ones we've made) won't be updated during training
        layer.trainable = False  
    return model

#plots a learning curve from a training history
def plot_learning_curve(history):
    plt.figure(figsize=(11, 6), dpi=100)
    plt.plot(history.history['loss'], 'o-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'o:', color='r', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss']), 10), range(1, len(history.history['loss']) + 1, 10))
    filename = 'LossCurve.png' 
    plt.savefig(filename) 
    
def plot_confusion_matrix(y_true, y_pred, classes, filename, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='black')
    fig.tight_layout()

    plt.savefig(filename)