import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_histogram(model_name, percent_accuracy):
    plt.figure()
    plt.hist(percent_accuracy, bins=100)
    plt.xlabel("Percent accuracy")
    plt.ylabel("Frequency")
    #plt.ylim(1,150) # for better comparison between models
    plt.axvline(percent_accuracy.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.title("Histogram of Percent Accuracy")
    plt.savefig("plots/{}/percent_accuracy_histogram.png".format(model_name))


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
    

VOXEL_COLORS = np.array(['brown','salmon','coral','midnightblue','olive',
                         'greenyellow','forestgreen','aquamarine', 
                         'mediumturquoise','teal','darkolivegreen','saddlebrown',
                         'dodgerblue','fuchsia','royalblue','darkorange',
                         'deeppink','blueviolet','plum','slategray',
                         'deeppink','black','navy','mediumslateblue','tomato',
                         'orangered','green'])


def _plot_event(fig, panel, event_id, event, title, colors=None):
    """ Only for internal use by plot_events """
    ax = fig.add_subplot(1, 4, panel, projection='3d')
    ax.axes.set_xlim3d(left=0, right=1)
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)
    if colors is None:
        colors = VOXEL_COLORS[event[:,3].astype(int)]
    '''
    Code to unscale, use as desired
    ax.axes.set_xlim3d(left=-255, right=255)
    ax.axes.set_ylim3d(bottom=-60, top=1000)
    ax.axes.set_zlim3d(bottom=-255, top=255)
    x_unscaled = (510*event[:,0])-255
    z_unscaled = (1060*event[:,2])-60
    y_unscaled = (510*event[:,1])-255
    ax.scatter3D(x_unscaled, z_unscaled, y_unscaled, color=colors, s=1)
    '''
    ax.scatter3D(event[:,0], event[:,2], event[:,1], color=colors, s=1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_zlabel('y [mm]')
    # plt.title('Event {} {}'.format(event_id, title))
    plt.title(title)
    
    
def plot_events(targets, predictions, data_file_stem, model_name):
    """
    Plots the original, shuffled, and unshuffled events along with the unshuffled
    events' accuracy. Five random events are chosen to be plotted.
    """
    
    # loading in data
    original_ds = np.load('{}{}'.format(data_file_stem, '_voxelated.npy'))
    shuffled_ds = np.load('{}{}'.format(data_file_stem, '_shuffled_voxels_only.npy'))
    base_voxels = np.load('{}{}'.format(data_file_stem, '_base_voxels.npy'))
    
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_event_nums = test_ds[:,:,5]
    test_event_ids = test_event_nums[:,0]

    voxel_bounds = np.load('voxel_data/voxel_bounds.npy')
    min_bounds = voxel_bounds[:, 0, :]
    
    
    # randomized event plotting. 5 random event IDs are chosen to plot. 
    # change the range to see fewer/more plots
    
    print("Number of events:", len(original_ds))
    print("Number of events in test set:", len(test_event_nums))

    for i in range(5):
        event_id = int(np.random.choice(test_event_ids)) # randomly chosen event ID
        
        fig = plt.figure(figsize=(17,7.5))

        # plot original events + shuffled events
        _plot_event(fig, 1, event_id, original_ds[event_id,:,:], 'Original Event')
        _plot_event(fig, 2, event_id, shuffled_ds[event_id,:,:], 'Shuffled Event')

        # plot predictions but with target colors (i.e., colored according to
        # the voxel of origin)
        translated_evt = base_voxels[event_id,:,:].copy()
        preds = predictions[i]
        translated_evt[:,:3] = translated_evt[:,:3] + min_bounds[preds] 
        _plot_event(fig, 3, event_id, translated_evt, 'Reconstructed Event')

        # plot predictions with hit/miss colors (misses = red, hits = blue)
        MISS_HIT_COLORS = np.array(['red', 'blue'])
        colors = MISS_HIT_COLORS[(targets[i,:] == predictions[i,:]).astype(int)]
        _plot_event(fig, 4, event_id, translated_evt, 'Reconstruction Accuracy', colors=colors)

        # plt.suptitle('Voxelated Event States Plotted', fontsize=25)
        plt.savefig('plots/{}/{}_voxels.png'.format(model_name, event_id))