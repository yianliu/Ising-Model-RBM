from Parameters import *
from MyFonts import *
from random import sample
import numpy as np
import os
import matplotlib.pyplot as plt

'''
# The code below plots the training set
for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join('Data', 'Training Data', file_name)
    all_samples = np.load(completeName)
    samples = sample(list(all_samples), 9)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows = 3, ncols = 3, figsize=(8,7))
    plt.style.use('grayscale')
    ax1.matshow(samples[0])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.matshow(samples[1])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.matshow(samples[2])
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4.matshow(samples[3])
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5.matshow(samples[4])
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6.matshow(samples[5])
    ax6.set_xticks([])
    ax6.set_yticks([])

    ax7.matshow(samples[6])
    ax7.set_xticks([])
    ax7.set_yticks([])

    ax8.matshow(samples[7])
    ax8.set_xticks([])
    ax8.set_yticks([])

    ax9.matshow(samples[8])
    ax9.set_xticks([])
    ax9.set_yticks([])

    fig.patch.set_facecolor('w')
    fig.suptitle('Training Data: T = ' + format(T, '.2f'), fontdict = myfont_l)
    figname = os.path.join('Plots', 'Configurations', 'Training T = ' + format(T, '.2f') + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig.savefig(figname, dpi = 400)

# The code below plots the RBM generated data
nH = 64
for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join('Data', 'RBM Generated Data', 'nH = ' + str(nH), file_name)
    all_samples = np.load(completeName)
    samples = sample(list(all_samples), 9)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows = 3, ncols = 3, figsize=(8,7))
    plt.style.use('grayscale')
    ax1.matshow(samples[0])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.matshow(samples[1])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.matshow(samples[2])
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4.matshow(samples[3])
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5.matshow(samples[4])
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6.matshow(samples[5])
    ax6.set_xticks([])
    ax6.set_yticks([])

    ax7.matshow(samples[6])
    ax7.set_xticks([])
    ax7.set_yticks([])

    ax8.matshow(samples[7])
    ax8.set_xticks([])
    ax8.set_yticks([])

    ax9.matshow(samples[8])
    ax9.set_xticks([])
    ax9.set_yticks([])

    fig.patch.set_facecolor('w')
    fig.suptitle('RBM Data: nH = ' + str(nH) + ' T = ' + format(T, '.2f'), fontdict = myfont_l)
    figname = os.path.join('Plots', 'Configurations', 'RBM nH = ' + str(nH) + ' T = ' + format(T, '.2f') + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig.savefig(figname, dpi = 400)'''

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (7, 4))
fig.suptitle('Ising Lattice Configurations Sampled Using the Metropolis-Hastings Algorithm')
plt.style.use('grayscale')
for i, ax in enumerate(fig.axes):
    T = T_range[i]
    T_name = 'T = ' + format(T, '.2f')
    file_name = T_name + '.npy'
    completeName = os.path.join('Data', 'Training Data', file_name)
    all_samples = np.load(completeName)
    samples = sample(list(all_samples), 1)
    ax.matshow(samples[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(T_name)
fig.patch.set_facecolor('w')
figname = os.path.join('Plots', 'Configurations', 'MCMC All Temps.jpg')
if os.path.isfile(figname):
   os.remove(figname)
fig.savefig(figname, dpi = 400)
