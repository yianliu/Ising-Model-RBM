from Parameters import *
from MyFonts import *
from random import sample
import numpy as np
import os
import matplotlib.pyplot as plt

numConfigurations = 36
numRows = 6
numCols = 6
# The code below plots the training set
for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join('Data', 'Training Data', file_name)
    all_samples = np.load(completeName)
    samples = sample(list(all_samples), numConfigurations)
    fig, axes = plt.subplots(nrows = numRows, ncols = numCols, figsize=(8,7))
    plt.style.use('grayscale')
    for axIndex, ax in enumerate(fig.axes):
        ax.matshow(samples[axIndex])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.patch.set_facecolor('w')
    fig.suptitle('Training Data: T = ' + format(T, '.2f'))
    figname = os.path.join('Plots', 'Configurations', 'Training T = ' + format(T, '.2f') + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig.savefig(figname, dpi = 400)

# The code below plots the RBM generated data
for nH in nH_list:
    for i in range(nt):
        T = T_range[i]
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeName = os.path.join('Data', 'RBM Generated Data', 'nH = ' + str(nH), file_name)
        all_samples = np.load(completeName)
        samples = sample(list(all_samples), numConfigurations)
        fig, axes = plt.subplots(nrows = numRows, ncols = numCols, figsize=(8,7))
        plt.style.use('grayscale')
        for axIndex, ax in enumerate(fig.axes):
            ax.matshow(samples[axIndex])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.patch.set_facecolor('w')
        fig.suptitle('RBM Data: nH = ' + str(nH) + ' T = ' + format(T, '.2f'))
        figname = os.path.join('Plots', 'Configurations', 'RBM nH = ' + str(nH) + ' T = ' + format(T, '.2f') + '.jpg')
        if os.path.isfile(figname):
           os.remove(figname)
        fig.savefig(figname, dpi = 400)


'''
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
fig.savefig(figname, dpi = 400)'''
