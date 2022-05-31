import cnn_functions as cnn
import numpy as np
import h5py
import tensorflow as tf
from sklearn.metrics import classification_report
import sys
from models.pointnet_model import pnet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
import random 



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
   
def evaluate1(targets, predictions):
    if config['class_type'] == 'BINARY':
        class_names = ['alpha', 'other']
    elif config['class_type'] == 'TERTIARY':
        class_names = ['Alpha', 'Neon', 'Proton']
    else:
        class_names = ['beam', 'two track', 'three track', 'four track', 'five track', 'six track']
        
    print(classification_report(targets, predictions, target_names=class_names))

    cnn.plot_confusion_matrix(targets, predictions, class_names, './ConfusionMatrix.png', title= config['class_type'] + ' Vanilla Pointnet Confusion Matrix ' + str(config['j']))
    print('Confusion matrix made!')
    
def evaluate2(features, targets, predictions):    
    
    #Event IDs within original_ds to first check if are in test then to be plotted
    rand_ints = [1752,493.0,1409.0,165.0,705.0,555.0,1507.0]
    #1752,493.0,1409.0,165.0,705.0,555.0,1507.0
    #1579.0,1927.0,1122.0,1625.0,678.0,99.0,1667.0,1408.0,1812.0,1752.0,890.0,1546.0,1161.0,794.0
#     rand_ints = []
#     for i in range(20):
#         rand_ints.append(random.randint(0,2425))
    
    original_ds = np.load(config['original_ds'])
    shuffled_ds = np.load(config['shuffled_ds'])
    base_voxels = np.load(config['base_voxels_ds'])
    
    test_event_nums = np.load(config['test_ds2'])
    test_event_nums = test_event_nums[:,:,5]
    
    event_accuracies = np.zeros((len(test_event_nums),))
    event_miss_rates = np.zeros((len(test_event_nums),))
    for i in range(len(test_event_nums)):
        event_num = int(test_event_nums[i,0])
        plot_flag = False
        for num in rand_ints:
            if num == event_num:
                plot_flag = True
                
        hits = 0
        misses = 0
        
        #If plotting events
        if plot_flag:
            datasets = [original_ds, shuffled_ds, base_voxels, base_voxels]
            
            colors = [
            'chocolate','grey','ivory',
            'wheat','olive','tan',
            'black','aqua','lightblue',
            'green','yellow','red',
            'blue','pink','indigo',
            'violet','chocolate','chartreuse',
            'cyan','fuchsia','azure',
            'navy','goldenrod','teal',
            'salmon','lime','darkblue',
            'r','b'
            ]
            fig = plt.figure(figsize=(17,7.5))
            for state in range(2):
                ds =  datasets[state]
                ax = fig.add_subplot(1, 4, state+1, projection='3d')
                ax.axes.set_xlim3d(left=0, right=1)
                ax.axes.set_ylim3d(bottom=0, top=1)
                ax.axes.set_zlim3d(bottom=0, top=1)
                #rand_int = random.randint(0,1999)
                evt = ds[event_num,:,:]
                for j,e in enumerate(evt):
                    x = e[0] #get x value of instance
                    y = e[1] #get y value of instance
                    z = e[2] #get z value of instance
                    clr = colors[int(e[3])]    #color based on real label
                    ax.scatter3D(x,z,y, color = clr, s = 1) 

                ax.set_xlabel('x [mm]')
                ax.set_ylabel('z [mm]')
                ax.set_zlabel('y [mm]')
                plt.title('Event ' +str(int(event_num))+ ' State ' +str(state+1))
                #plt.savefig('./' + str(1234) + '/' +str(int(event_num))+'/' +str(int(event_num))+ '_' +str(state+1)+ '_Voxels.png')
            
            #Plotting original Event w/ hits and misses for each point's labels
            voxel_bounds = np.load(config['voxel_bounds'])
            for state in range(2):
                ax = fig.add_subplot(1, 4, state+3, projection='3d')
                ax.axes.set_xlim3d(left=0, right=1) 
                ax.axes.set_ylim3d(bottom=0, top=1) 
                ax.axes.set_zlim3d(bottom=0, top=1)

                ds = datasets[state+2]
                evt = ds[event_num,:,:]
                for j,e in enumerate(evt):
                    x = e[0] #get x value of instance
                    y = e[1] #get y value of instance
                    z = e[2] #get z value of instance
                    pred = predictions[i,j]
                    targ = targets[i,j]
                    min_bounds = voxel_bounds[pred,0]
                    x += min_bounds[0]
                    y += min_bounds[1]
                    z += min_bounds[2]
                    if state == 0:
                        if pred != targ:
                            misses += 1
                            clr = colors[int(targ)]
                        else:
                            hits += 1
                            clr = colors[int(targ)]
                    else:
                        if pred != targ:
                            misses += 1
                            clr = colors[-2]
                        else:
                            hits += 1
                            clr = colors[-1]
                    ax.scatter3D(x,z,y, color = clr, s = 1)
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('z [mm]')
                ax.set_zlabel('y [mm]')
                if state == 1:
                    miss_patch = patches.Patch(color=colors[-2], label = 'miss')
                    hit_patch = patches.Patch(color=colors[-1], label = 'hit')
                    plt.legend(handles=[miss_patch,hit_patch])
                plt.title('Event ' +str(int(event_num))+ ' State ' +str(state+3))
                #plt.savefig('./' + str(1234) + '/' +str(int(event_num))+'/' +str(int(event_num))+ '_' +str(state+3)+ '_Voxels.png')
                
                
            
            plt.suptitle('Voxelated Event States Plotted', fontsize=25)
            plt.savefig('./' + str(1234) + '/' +str(event_num)+'_Voxels.png')
            print('Events plotted!')
            
        #If Not plotting Events   
        else:
            for j in range(512):
                pred = predictions[i,j]
                targ = targets[i,j]
                if pred != targ:
                    misses += 1
                else:
                    hits += 1
                    
        event_accuracies[i] = hits/(hits+misses)
        event_miss_rates[i] = misses/(hits+misses)
        
        
    mean_accuracy = np.mean(event_accuracies)
    mean_miss_rate = np.mean(event_miss_rates)
    print('Average Percent Correct: ' + str(mean_accuracy))
    print('Average Miss Rate: ' + str(mean_miss_rate))
    
if __name__ == '__main__':
    #Note the data, data2, & data3. Change to your data locations
    config = {
    #For Whole Event Classification
    'test_points' : 'data/Mg22_size512test_convertXYZ.npy',
    'test_labels' : 'data/Mg22_size512_track_labels_test_convertXYZ.npy',
    #For Semantic Segmentation Classification
    'test_ds1' : 'data2/Mg22_size{}test_convert{}.npy',
    #For Voxxle Shuffle Method
    'test_ds2' : 'data3/Mg22_size512test.npy',
    'original_ds' : 'data3/Mg22_size512_voxelated.npy',
    'shuffled_ds' : 'data3/Mg22_size512_shuffled_voxels.npy',
    'base_voxels_ds' : 'data3/Mg22_size512_base_voxels.npy',
    'voxel_bounds' : 'data3/voxel_bounds.npy',
    #Change for your data/model specifications
    'num_points' : 512,
    'batch_size' : 5,
    'projection' : 'XYZ',
    #Command Line Arguements
    'j' : sys.argv[1], #Default to 1234                    #Typically Slurm Job #, used for directory to save results
    'class_type' : sys.argv[2], #Default to to 'BINARY'
    'sem_seg_flag' : sys.argv[3], #Default to True         #True if using semantic segmentation, False for whole event classififaction
    'num_classes' : sys.argv[4], #Default to 2
    'voxel_shuffle_flag' : sys.argv[5] #Default to False   #True to evaluate Voxel Shuffle Method
    }
    
    #Voxle Shuffle Method Evaluation using Vailla Pointnet Model
    if config['voxel_shuffle_flag']:
        model = build_model()
        test_features, test_targets = load_data2(config['test_ds2'])
        predictions = make_predictions2(model, test_features)
        evaluate2(test_features, test_targets, predictions)
        
    #Just Vanilla Pointnet Evaluation
    else:
        #Semantic Segmentation Classification
        if config['sem_seg_flag']:
            model = build_model()
            set1 = config['test_ds1'].format(config['num_points'], str(1) + config['projection'])
            set2 = config['test_ds1'].format(config['num_points'], str(2) + config['projection'])

            test_features1, test_targets1 = load_data2(set1)
            test_features2, test_targets2 = load_data2(set2)
            predictions1 = make_predictions2(model, test_features1)
            predictions2 = make_predictions2(model, test_features2)

            targets, predictions = process(test_targets1, test_targets2, predictions1, predictions2)
            evaluate1(targets, predictions)
            
        #Whole Event Classification
        else:
            model = build_model()
            test_features, test_targets = load_data1(config['test_points'], config['test_labels'])
            predictions = make_predictions1(model, test_features)
            evaluate1(test_targets, predictions)
    
    
    
    
