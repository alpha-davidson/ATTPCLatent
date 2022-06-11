import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
import sys

sample_size = 512
color = [(.2,.92,.75), (.6,.2,1), (0,0,0), (.92,.37,.53), 'r','y', 'm','c'] #characters corresponding to color
    
j = sys.argv[1]
class_type = sys.argv[2]

test_ev_lens = np.load('./data2/Mg22_4-tracktestevent_lengths.npy')

def plot_events(ds_path, rows, is_pred):
    test_set =  np.load(ds_path) 
    fig = plt.figure(figsize=(17,7.5))
    for n in range(rows*5):
        conflicts = 0
        errors = 0
        ax = fig.add_subplot(rows, 5, n+1, projection='3d')
        evt = test_set[n,:test_ev_lens[n],:]
        for i,e in enumerate(evt):
            x = e[0] #get x value of instance
            y = e[1] #get y value of instance
            z = e[2] #get z value of instance
            if is_pred:
                if int(e[-1]) == 1:
                    conflicts += 1
                    clr = color[2]
                elif int(e[7]) != int(e[3]):
                    clr = color[3]
                    errors += 1
                else: 
                    clr = color[int(e[7])]    #color based on prediction
            else:
                clr = color[int(e[3])]    #color based on real label
            ax.scatter3D(x,z,y, color = clr, s = 1) 
 
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_zlabel('y [mm]')
        
        alpha = patches.Patch(color=color[0], label = 'alpha')
        non = patches.Patch(color=color[1], label = 'non-alpha')
        conflict = patches.Patch(color=color[2], label = 'conflicting prediction')
        error = patches.Patch(color=color[3], label = 'incorrect prediction')

        if is_pred:
            plt.title('Event {} \n'.format(int(evt[0,4])) + str(evt.shape[0]) + ' points, ' + str(conflicts) + ' conflicts \n {} errors'.format(errors))
            plt.legend(handles=[alpha, non, conflict, error], fontsize='xx-small')

        else:
            plt.title('Event {}'.format(int(evt[0,4])))
            plt.legend(handles=[alpha, non])


            
    if is_pred:
        plt.suptitle(class_type + ' Predicted Events ' + str(j), fontsize=25)
        plt.savefig('./' + str(j) + '/PredictedEvents.png')
        print('Predictions plotted!')
    else:
        plt.suptitle(class_type + ' Real Events ' + str(j), fontsize=25)
        plt.savefig('./' + str(j) + '/RealEvents.png')
        print('Events plotted!')


if __name__ == '__main__':
    
    ds_path = './data2/Mg22_whole_4-track' + class_type + '_test.npy'
    
    plot_events(ds_path, 2, False)
    plot_events(ds_path, 2, True)
