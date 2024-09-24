import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import datetime

def plot_histogram(model_name, percent_accuracy, ckpt_name):
    plt.figure()
    plt.hist(percent_accuracy, bins=100)
    plt.axvline(percent_accuracy.mean(), color='black', linestyle='dashed', linewidth=1)  # adding mean line and value to histogram
    min_ylim, max_ylim = plt.ylim()
    plt.text(percent_accuracy.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(percent_accuracy.mean()))
    #adding histogram labels
    plt.xlabel("Percent accuracy")
    plt.ylabel("Frequency")
    plt.title("Histogram of Percent Accuracy")
    plt.savefig("Mg22_plots/{}/{}/percent_accuracy_histogram.png".format(model_name,ckpt_name))

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

def plot_events(targets, predictions, data_file_stem, model_name, ckpt_name):
    """
    Plots the original, shuffled, and unshuffled events along with the unshuffled
    events' accuracy. This function currently randomly plots 5 events from the 
    test event dataset.
    """

    # loading in data
    original_ds = np.load('{}{}'.format(data_file_stem, '_voxelated.npy'))
    shuffled_ds = np.load('{}{}'.format(data_file_stem, '_shuffled_voxels_only.npy'))
    base_voxels = np.load('{}{}'.format(data_file_stem, '_base_voxels.npy'))

    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_event_nums = test_ds[:,:,5]
    
    print("Number of events:", len(original_ds))
    print("Number of events in test set:", len(test_event_nums))

    # TODO: remove hardcoding
    voxel_bounds = np.load('Mg22_voxel_data/voxel_bounds.npy')
    min_bounds = voxel_bounds[:, 0, :]
    

    os.mkdir('Mg22_plots/{}/{}/'.format(model_name, ckpt_name))

    for j in range(5):
        i = np.random.randint(len(test_event_nums[:,0]))
        event_id = int(test_event_nums[i,0])
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
        plt.savefig('Mg22_plots/{}/{}/{}_voxels.png'.format(model_name,ckpt_name, event_id))
            
    
def plot_identity_events(targets, predictions, data_file_stem, model_name,ckpt_name):
    """
    When called, this function only plots the events that were left "unshuffled". 
    The first two graphs should be the same. These are called identity events. 
    """
    
    # loading in data
    original_ds = np.load('{}{}'.format(data_file_stem, '_voxelated.npy'))
    shuffled_ds = np.load('{}{}'.format(data_file_stem, '_shuffled_voxels_only.npy'))
    base_voxels = np.load('{}{}'.format(data_file_stem, '_base_voxels.npy'))
    
    voxel_bounds = np.load('Mg22_voxel_data/voxel_bounds.npy')
    min_bounds = voxel_bounds[:, 0, :]
    
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_event_nums = test_ds[:,:,5]
    
    print("Number of events:", len(original_ds))
    print("Number of events in test set:", len(test_event_nums))
    
    timestamp = timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    os.makedirs('Mg22_plots/{}/{}/identity_events'.format(timestamp, ckpt_name)) 

    # finding the identity events
    for i in range(len(targets)):
        event_id = int(test_event_nums[i,0])
        OG_test = original_ds[event_id,:,:]
        shuffled_test = shuffled_ds[event_id,:,:]
        for j in range(len(OG_test[0])):
            for k in range(len(OG_test[1])):
                if OG_test[j,k] == shuffled_test[j,k]:
                    plot = True
                else:
                    plot = False
        if plot:
            fig = plt.figure(figsize=(17,7.5))
            
            # plot OG and shuffled events
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

            plt.savefig('Mg22_plots/{}/{}/identity_events/{}_voxels.png'.format(model_name, ckpt_name, event_id))
    
    
def plot_zero_one_bins(targets, predictions, data_file_stem, model_name, ckpt_name):
    """ 
    This function only plots the graphs that had 0% accuracy and 100% 
    accuracy, or the events that went into the '0 bin' and '1 bin' on the histogram. 
    """

    # loading in data
    original_ds = np.load('{}{}'.format(data_file_stem, '_voxelated.npy'))
    shuffled_ds = np.load('{}{}'.format(data_file_stem, '_shuffled_voxels_only.npy'))
    base_voxels = np.load('{}{}'.format(data_file_stem, '_base_voxels.npy'))
    
    voxel_bounds = np.load('Mg22_voxel_data/voxel_bounds.npy')
    min_bounds = voxel_bounds[:, 0, :]
    
    test_ds = np.load('{}{}'.format(data_file_stem, 'test.npy'))
    test_event_nums = test_ds[:,:,5]
    
    count = 0
    zero_bin = []
    one_bin = []

    os.makedirs('Mg22_plots/{}/{}/0_bin'.format(model_name, ckpt_name))
    os.makedirs('Mg22_plots/{}/{}/1_bin'.format(model_name, ckpt_name))

    for i in range(len(targets)):
        
        # finds the events from the zero bin
        if np.mean(targets[i,:] == predictions[i]) == 0.0:
            event_id = int(test_event_nums[i,0])          
            fig = plt.figure(figsize=(17,7.5))
            zero_bin.append(original_ds[event_id,:,:])

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
            
            plt.savefig('Mg22_plots/{}/{}/0_bin/{}_voxels.png'.format(model_name, ckpt_name, event_id))

        # finds the events from the one bin
        elif np.mean(targets[i,:] == predictions[i]) == 1.0:    # 100% accuracy
            event_id = int(test_event_nums[i,0])          
            fig = plt.figure(figsize=(17,7.5))
            one_bin.append(original_ds[event_id,:,:])

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
            plt.savefig('Mg22_plots/{}/{}/1_bin/{}_voxels.png'.format(model_name,ckpt_name,event_id))
            
    # saving the data for the 0% and 100% accuracy events        
    np.save('Mg22_plots/{}/{}/0_bin/0_data'.format(model_name, ckpt_name), zero_bin)
    np.save('Mg22_plots/{}/{}/1_bin/1_data'.format(model_name, ckpt_name), one_bin)