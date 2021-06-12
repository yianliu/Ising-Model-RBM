# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:05:31 2021

@author: Yian Liu
"""

import numpy as np
import copy
import os
from Parameters import *

# Hamiltonian of a simple 2D lattice with configuration spins (ndarray) and coupling coefficient J = 1
def H(spins):
    coup = 0
    (m, n) = spins.shape
    for i in range(m - 1):
        for j in range(n - 1):
            coup += spins[i, j] * spins[i + 1, j] + spins[i, j] * spins[i, j + 1]
        coup += spins[i, n - 1] * spins[i + 1, n - 1]
    for j in range(n - 1):
        coup += spins[n - 1, j] * spins[n - 1, j + 1]
    return - coup

# Unormalised probability distribution for spins at temperature T
def prob(spins, T):
    p = np.exp(- H(spins) / T)
    return p

# Random spin configuration generator with dimensions n*n
def rdspins(n):
    state = 2 * np.random.randint(2, size=(n, n)) - 1
    return state

# dH calculates the change in Hamiltonian with the (i,j) spin flipped
def dH(spins, i, j):
    s = spins[i,j]
    (m, n) = spins.shape
    H, H_new = 0, 0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        inew = i + di
        jnew = j + dj
        if inew in range(m) and jnew in range(n):
            H -= s * spins[i + di, j + dj]
            H_new += s * spins[i + di, j + dj] # because s because -s and the Hamiltonian has a "-" sign
    return H_new - H

# Metropolis-Hastings algorithm for MCMC method
def MH(spins, T, steps):
    (h, w) = spins.shape
    for step in range(steps):
        for i in range(h):
            for j in range(w):
                dh = dH(spins, i, j)
                if dh <= 0:
                    spins[i, j] = - spins[i, j]
                else:
                    prob = np.exp(- dh / T)
                    if np.random.rand() < prob:
                        spins[i, j] = - spins[i, j]
    return spins

# MCsample is a function whichh generates sz configurations which mimics the
# behaviour of spins of a n*n lattice at temperature T. The first configuration
# is sampled after eqsteps and after that the interval between two configurations
# is mcsteps
def MCsample(n, T, sz, eqsteps, mcsteps):
    inits = rdspins(n)
    spins = MH(inits, T, eqsteps)
    snapshots = [copy.deepcopy(spins)]
    for i in range(sz - 1):
        spins = MH(spins, T, mcsteps)
        snapshots.append(copy.deepcopy(spins))
        print(i)
    return snapshots

# data_path includes the name of the directory where the dataset will be stored
data_path = 'Training Data'

# The following step generates datasets of size sz each and stores them in the
# "Training Data" folder with the corresponding temperature as the filename
for T in T_range:
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join('Data', data_path, file_name)
    samples = MCsample(n, T, sz, eqsteps, mcsteps)
    np.save(completeName, samples)
