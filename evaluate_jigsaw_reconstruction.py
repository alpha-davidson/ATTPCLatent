import plotting
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from pointnet_model import pnet
import click
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@click.command()
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.argument('model-file-stem')
@click.argument('data-file-stem')
def evaluate(num_points, num_classes, model_file_stem, data_file_stem):
    """
    Sample invocation:
        python3 evaluate_jigsaw_reconstruction.py --num-classes 27 models/2022-06-01-16:45:13/weights \
          voxel_data/Mg22_size512
    """
    # build model
    model = pnet(sem_seg_flag=True, num_points=num_points, num_classes=num_classes)
    model.load_weights(model_file_stem)

    # load test data
    BATCH_SIZE = 32
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_features = tf.data.Dataset.from_tensor_slices(test_ds[:, :, :3]).batch(BATCH_SIZE)
    test_labels = test_ds[:, :, 3]
    
    # make predictions
    predicted_probabilities = model.predict(test_features)
    predictions = np.argmax(predicted_probabilities, axis=2)

    # evaluate results
    print('Mean accuracy: {}'.format(np.mean(test_labels == predictions)))
    plot_events(test_labels, predictions, data_file_stem)


VOXEL_COLORS = np.array(['chocolate','grey','ivory','wheat','olive','tan',
                         'black','aqua','lightblue','green','yellow','red',
                         'blue','pink','indigo','violet','chocolate','chartreuse',
                         'cyan','fuchsia','azure','navy','goldenrod','teal',
                         'salmon','lime','darkblue'])
def plot_event(fig, panel, event_id, event, title):
    ax = fig.add_subplot(1, 4, panel, projection='3d')
    ax.axes.set_xlim3d(left=0, right=1)
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)
    colors = VOXEL_COLORS[event[:,3].astype(int)]
    ax.scatter3D(event[:,0], event[:,2], event[:,1], color=colors, s=1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_zlabel('y [mm]')
    plt.title('Event {} {}'.format(event_id, title))

    
def plot_events(targets, predictions, data_file_stem):
    # 'r', 'b'
    # Event IDs within original_ds to plot, if they occur in the test set; these 
    # were hand-picked for offering good visualization of outcomes. May later
    # want to change to randomly picked IDs from test set. 
    events_to_plot = {1752, 493, 1409, 165, 705, 555, 1507, 1579, 1927,
                      1122, 1625, 678, 99, 1667, 1408, 1812, 1752, 890, 
                      1546, 1161, 794}
                          
    original_ds = np.load('{}{}'.format(data_file_stem, '_voxelated.npy'))
    shuffled_ds = np.load('{}{}'.format(data_file_stem, '_shuffled_voxels.npy'))
    base_voxels = np.load('{}{}'.format(data_file_stem, '_base_voxels.npy'))
    
    test_event_nums = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_event_nums = test_event_nums[:,:,5]
    
    # todo: remove hardcoding
    voxel_bounds = np.load('voxel_data/voxel_bounds.npy')
    
    for i in range(len(test_event_nums)):
        event_id = int(test_event_nums[i,0])
        if event_id in events_to_plot:            
            fig = plt.figure(figsize=(17,7.5))
            
            # plot original events + shuffled eventss
            plot_event(fig, 1, event_id, original_ds[event_id,:,:], 'original')
            plot_event(fig, 2, event_id, shuffled_ds[event_id,:,:], 'shuffled')
            
            # plot predictions + hit/miss colored predictions
            # plot_event(fig, 3, event_id, base_voxels[event_id,:,:], 'predictions')
            

#             for state in range(2):

#                 ds = datasets[state+2]
#                 evt = ds[event_num,:,:]
#                 for j,e in enumerate(evt):
#                     x = e[0] #get x value of instance
#                     y = e[1] #get y value of instance
#                     z = e[2] #get z value of instance
#                     pred = predictions[i,j]
#                     targ = targets[i,j]
#                     min_bounds = voxel_bounds[pred,0]
#                     x += min_bounds[0]
#                     y += min_bounds[1]
#                     z += min_bounds[2]
#                     if state == 0:
#                         if pred != targ:
#                             misses += 1
#                             clr = colors[int(targ)]
#                         else:
#                             hits += 1
#                             clr = colors[int(targ)]
#                     else:
#                         if pred != targ:
#                             misses += 1
#                             clr = colors[-2]
#                         else:
#                             hits += 1
#                             clr = colors[-1]
#                     ax.scatter3D(x,z,y, color = clr, s = 1)
#                 ax.set_xlabel('x [mm]')
#                 ax.set_ylabel('z [mm]')
#                 ax.set_zlabel('y [mm]')
#                 if state == 1:
#                     miss_patch = patches.Patch(color=colors[-2], label = 'miss')
#                     hit_patch = patches.Patch(color=colors[-1], label = 'hit')
#                     plt.legend(handles=[miss_patch,hit_patch])
#                 plt.title('Event ' +str(int(event_num))+ ' State ' +str(state+3))
                
                
            
            plt.suptitle('Voxelated Event States Plotted', fontsize=25)
            plt.savefig(str(event_id)+'_Voxels.png')
            print('Events plotted!')
            
    
if __name__ == '__main__':
    evaluate()
