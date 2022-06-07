import plotting
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from pointnet_model import pnet
import click

# 

@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('model-file-stem')
@click.argument('data-file-path')
def evaluate(num_points, num_classes, model_file_stem, data_file_path):
    """
    Sample invocation:
        python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2022-06-01-16:45:13/weights \
          voxel_data/Mg22_size512test.npy
    """
    # build model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)
    model.load_weights(model_file_stem)

    # load test data
    BATCH_SIZE = 32
    test_ds = np.load(data_file_path)
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :3]).batch(BATCH_SIZE)
    test_labels = test_ds[:, :, 3]
    
    # make predictions
    predicted_probabilities = model.predict(test_features)
    predictions = np.argmax(predicted_probabilities, axis=2)

    # evaluate results
    evaluate2(test_features, test_labels, predictions)


def evaluate2(features, targets, predictions):    
    """
        'test_ds2' : 'data3/Mg22_size512test.npy',
    -    'original_ds' : 'data3/Mg22_size512_voxelated.npy',
    -    'shuffled_ds' : 'data3/Mg22_size512_shuffled_voxels.npy',
    -    'base_voxels_ds' : 'data3/Mg22_size512_base_voxels.npy',
    -    'voxel_bounds' : 'data3/voxel_bounds.npy',
    """
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
    evaluate()
