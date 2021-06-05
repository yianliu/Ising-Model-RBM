 # -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:05:31 2021

@author: Yian Liu
"""

import numpy as np
import copy

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
    return snapshots

# T_range contains nt evenly spaced temperature points between 1 and 3.5
nt = 20
T_range = np.linspace(1, 3.5, nt)

# Below are the parameters (n: square lattice length; sz: size of dataset at
# each temperaturevalue; eqsteps: steps taken to reach equilibrium; mcsteps:
# number of intervals between configurations) used to build a list confgs where
# each element of the list is sz configurations at the corresponding temperature
confgs = []
n = 6
sz = 200
eqsteps = 100
mcsteps = 100

for i in range(nt):
    T = T_range[i]
    samples = MCsample(n, T, sz, eqsteps, mcsteps)
    confgs.append(copy.deepcopy(samples))

np.save("configurations.npy", confgs)
