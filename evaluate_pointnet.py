import cnn_functions as cnn
import numpy as np
import h5py
import tensorflow as tf
from sklearn.metrics import classification_report
import sys
from models.pointnet_model import pnet



def build_model():
    model = pnet(config['sem_seg_flag'], config['num_points'], int(config['num_classes']))
    model.load_weights('./logs/pnet_1/model/weights')
    return model

def load_data1(points_path, label_path):
    test_points = np.load(points_path)
    test_features = tf.data.Dataset.from_tensor_slices(test_points[:, :, :3]).batch(config['batch_size'], drop_remainder=True)
    test_targets = np.load(label_path)
    return test_features, test_targets

def load_data2(ds_path):
    test_ds = np.load(ds_path)
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :3]).batch(config['batch_size'], drop_remainder=True)
    test_targets = test_ds[:, :, 3]
    return test_features, test_targets

def make_predictions1(model, features):
    predicted_probabilities = model.predict(features)
    predictions = np.argmax(predicted_probabilities, axis=1)
    return predictions

def make_predictions2(model, features):
    predicted_probabilities = model.predict(features)
    predictions = np.argmax(predicted_probabilities, axis=2)
    return predictions

def process(real1, real2, pred1, pred2):
    if real1.shape[0] > pred1.shape[0]:
        diff = real1.shape[0] - pred1.shape[0]
        real1 = real1[:-diff,:]
        real2 = real2[:-diff,:]
        
    targets = np.zeros(2*real1.size)
    targets[:real1.size] = real1.flatten()
    targets[real1.size:] = real2.flatten()
    
    predictions = np.zeros(2*pred1.size)
    predictions[:pred1.size] = pred1.flatten()
    predictions[pred1.size:] = pred2.flatten()
    
    return targets, predictions
   
def evaluate(targets, predictions):
    if config['class_type'] == 'BINARY':
        class_names = ['alpha', 'other']
    elif config['class_type'] == 'TERTIARY':
        class_names = ['Alpha', 'Neon', 'Proton']
    else:
        class_names = ['beam', 'two track', 'three track', 'four track', 'five track', 'six track']
    
    print(classification_report(targets, predictions, target_names=class_names))

    cnn.plot_confusion_matrix(targets, predictions, class_names, './ConfusionMatrix.png', title= config['class_type'] + ' Vanilla Pointnet Confusion Matrix ' + str(config['j']))
    print('Confusion matrix made!')
    
    
if __name__ == '__main__':
    config = {
    'test_points' : 'data/Mg22_size512test_convertXYZ.npy',
    'test_labels' : 'data/Mg22_size512_track_labels_test_convertXYZ.npy',
    'test_ds' : 'data2/Mg22_size{}test_convert{}.npy',
    'num_points' : 512,
    'batch_size' : 16,
    'num_classes' : sys.argv[4],
    'sem_seg_flag' : sys.argv[3],
    'j' : sys.argv[1],
    'class_type' : sys.argv[2],
    'projection' : 'XYZ'
    }
    
    if config['sem_seg_flag']:
        model = build_model()
        set1 = config['test_ds'].format(config['num_points'], str(1) + config['projection'])
        set2 = config['test_ds'].format(config['num_points'], str(2) + config['projection'])

        test_features1, test_targets1 = load_data2(set1)
        test_features2, test_targets2 = load_data2(set2)
        predictions1 = make_predictions2(model, test_features1)
        predictions2 = make_predictions2(model, test_features2)

        targets, predictions = process(test_targets1, test_targets2, predictions1, predictions2)

        evaluate(targets, predictions)
    else:
        model = build_model()
        test_features, test_targets = load_data1(config['test_points'], config['test_labels'])
        predictions = make_predictions1(model, test_features)
        evaluate(test_targets, predictions)
    
    
    
    
    
    
