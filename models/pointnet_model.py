import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

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
    
def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 16)
    x = conv_bn(x, 32)
    x = conv_bn(x, 256)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 128)
    x = dense_bn(x, 64)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def pnet(sem_seg_flag, num_points, num_classes):
    # Since current number of classes is about 1/4 of pointnet, used 1/4 of size 
    # for all layers except initial 3 for input (x,y,z)    
    inputs = keras.Input(shape=(num_points, 3))
    a = tnet(inputs, 3)
    b = conv_bn(a, 32)
    c = conv_bn(b, 32)
    d = tnet(c, 32)
    e = conv_bn(d, 32) 
    f = conv_bn(e, 64)
    g = conv_bn(f, 512)
    x = layers.GlobalMaxPooling1D()(g)

    if sem_seg_flag:
        #Current uses Fig 9 archituecture, can change to Fig 2 later
        long = layers.Concatenate(axis=2)([b,c])
        long = layers.Concatenate(axis=2)([long,d])
        long = layers.Concatenate(axis=2)([long,e])
        long = layers.Concatenate(axis=2)([long,f])
        x = tf.expand_dims(x, axis=1)
        x = tf.repeat(x, repeats=512, axis=1)
        long = layers.Concatenate(axis=2)([long,x])
        x = conv_bn(x, 256)
        x = conv_bn(long, 128)
        x = conv_bn(x, 128)
        x = conv_bn(x, 64)
        x = layers.Dropout(0.3)(x)
        x = conv_bn(x, 64)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
    else:
        x = dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()
        
    return model