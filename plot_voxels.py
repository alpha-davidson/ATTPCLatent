import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
import sys
import random

def plot_events(event):
    fig = plt.figure(figsize=(17,7.5))
    for state in range(2):
        path = 'ds_path' + str(state+1)
        ds =  np.load(config[path])
        ax = fig.add_subplot(1, 2, state+1, projection='3d')
        ax.axes.set_xlim3d(left=0, right=1)
        ax.axes.set_ylim3d(bottom=0, top=1)
        ax.axes.set_zlim3d(bottom=0, top=1)
        #rand_int = random.randint(0,1999)
        evt = ds[event,:,:]
        voxels = np.zeros((27,))
        for i,e in enumerate(evt):
            x = e[0] #get x value of instance
            y = e[1] #get y value of instance
            z = e[2] #get z value of instance
            clr = color[int(e[3])]    #color based on real label
            voxels[int(e[3])] += 1
            ax.scatter3D(x,z,y, color = clr, s = 1) 

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_zlabel('y [mm]')
        
        plt.title('Event ' +str(int(evt[0,5]))+ ' State ' +str(state+1))
#         voxels = []
#         for i in range(27):
#             voxels.append(patches.Patch(color=color[i], label = str(i)))
#         plt.legend(handles=voxels)
            
    plt.suptitle('Voxelated Event States Plotted', fontsize=25)
    plt.savefig('./' + str(1234) + '/' +str(event)+'_Voxels.png')
    print('Events plotted!')


if __name__ == '__main__':
    config = {
    'ds_path1' : './data3/Mg22_size512_voxelated.npy',
    'ds_path2' : './data3/Mg22_size512_shuffled_voxels.npy'
    }
    color = [
    'chocolate','grey','ivory',
    'wheat','olive','tan',
    'black','aqua','lightblue',
    'green','yellow','red',
    'blue','pink','indigo',
    'violet','chocolate','chartreuse',
    'cyan','fuchsia','azure',
    'navy','goldenrod','teal',
    'salmon','lime','darkblue'
    ]
    #ind = [516, 1076, 1172, 1477, 2025]
    ind = [1752]
    #945, 50
    for event in ind:
        plot_events(event)
