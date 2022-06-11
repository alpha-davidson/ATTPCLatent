import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_learning_curve(history, filename):
    plt.figure(figsize=(11, 6), dpi=100)
    plt.plot(history.history['loss'], 'o-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'o:', color='r', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss']), 10), range(1, len(history.history['loss']) + 1, 10))
    plt.yscale('log')
    plt.savefig(filename) 
    
    
def plot_confusion_matrix(y_true, y_pred, classes, filename, title='Confusion Matrix', cmap=plt.cm.Blues):
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
    

VOXEL_COLORS = np.array(['chocolate','grey','ivory','wheat','olive','tan',
                         'black','aqua','lightblue','green','yellow','red',
                         'blue','pink','indigo','violet','chocolate','chartreuse',
                         'cyan','fuchsia','azure','navy','goldenrod','teal',
                         'salmon','lime','darkblue'])
def _plot_event(fig, panel, event_id, event, title, colors=None):
    """ Only for internal use by plot_events """
    ax = fig.add_subplot(1, 4, panel, projection='3d')
    ax.axes.set_xlim3d(left=0, right=1)
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)
    if colors is None:
        colors = VOXEL_COLORS[event[:,3].astype(int)]
    ax.scatter3D(event[:,0], event[:,2], event[:,1], color=colors, s=1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_zlabel('y [mm]')
    plt.title('Event {} {}'.format(event_id, title))

    
def plot_events(targets, predictions, data_file_stem):
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
    min_bounds = voxel_bounds[:, 0, :]
    
    for i in range(len(test_event_nums)):
        event_id = int(test_event_nums[i,0])
        if event_id in events_to_plot:            
            fig = plt.figure(figsize=(17,7.5))
            
            # plot original events + shuffled events
            _plot_event(fig, 1, event_id, original_ds[event_id,:,:], 'original')
            _plot_event(fig, 2, event_id, shuffled_ds[event_id,:,:], 'shuffled')
            
            # plot predictions but with target colors (i.e., colored according to
            # the voxel of origin)
            translated_evt = base_voxels[event_id,:,:].copy()
            preds = predictions[i]
            translated_evt[:,:3] = translated_evt[:,:3] + min_bounds[preds]
            _plot_event(fig, 3, event_id, translated_evt, 'predictions')
            
            # plot predictions with hit/miss colors (misses = red, hits = blue)
            MISS_HIT_COLORS = np.array(['red', 'blue'])
            colors = MISS_HIT_COLORS[(targets[i,:] == predictions[i,:]).astype(int)]
            _plot_event(fig, 4, event_id, translated_evt, 'hits (blue)/misses (red)', colors=colors)

            plt.suptitle('Voxelated Event States Plotted', fontsize=25)
            plt.savefig('{}_voxels.png'.format(event_id))
