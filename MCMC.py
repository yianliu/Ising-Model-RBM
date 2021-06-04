 # -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:05:31 2021

@author: Yian Liu
"""

import numpy as np
import matplotlib.pyplot as plt
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

# Below are the functions for different thermodynamic properties

# Magnetisation per spin
M = np.zeros((nt,sz)) # for all configurations
M_mean = np.zeros(nt)
M_err = np.zeros(nt)
M2 = np.zeros((nt,sz)) # magnetisation per spin squared
M2_mean = np.zeros(nt)
M2_err = np.zeros(nt)

# Energy per spin
E = np.zeros((nt,sz)) # for all configurations
E_mean = np.zeros(nt)
E_err = np.zeros(nt)
E2 = np.zeros((nt,sz)) # energy per spin sqaured
E2_mean = np.zeros(nt)
E2_err = np.zeros(nt)

# Specific Heat
Cv_mean = np.zeros(nt)
Cv_err = np.zeros(nt)

# Magnetic Susceptibility
X_mean = np.zeros(nt)
X_err = np.zeros(nt)

# Mag is a function which takes a configuration and returns the total magnetisation
# of the entire lattice
def Mag(spins):
    mag = np.sum(spins)
    return mag

# error is a function which calculates the standard error of a list of values
def error(lst):
    return np.std(lst, ddof = 1) / np.sqrt(np.size(lst))

for i in range(nt):
    T = T_range[i]
    samples = MCsample(n, T, sz, eqsteps, mcsteps)
    confgs.append(copy.deepcopy(samples))

np.save("configurations.npy", confgs)
confgs = np.load("configurations.npy")
# Below are the calculations of thermodynamic properties
for i in range(nt):
    T = T_range[i]
    samples = confgs[i]
    for j in range(sz):
        spins = samples[j]
        N = spins.size
        m = abs(Mag(spins))/N
        h = H(spins)/N
        M[i, j] = m
        E[i, j] = h
        M2[i, j] = m*m
        E2[i, j] = h*h
    E_mean[i] = np.mean(E[i])
    E_err[i] = error(E[i])
    E2_mean[i] = np.mean(E2[i])
    E2_err[i] = error(E2[i])
    M_mean[i] = np.mean(M[i])
    M_err[i] = error(M[i])
    M2_mean[i] = np.mean(M2[i])
    M2_err[i] = error(M2[i])
    Cv_mean[i] = (E2_mean[i] - E_mean[i]**2) / (N * T**2)
    Cv_err[i] = (E2_err[i] - 2 * E_mean[i] * E_err[i]) / (N * T**2)
    X_mean[i] = (M2_mean[i] - M_mean[i]**2) / (N * T)
    X_err[i] = (M2_err[i] - 2 * M_mean[i] * M_err[i]) / (N * T)

plt.xlabel('Temperature')
plt.ylabel('E')
plt.errorbar(T_range, E_mean, yerr=E_err, xerr=None)
plt.show()

plt.xlabel('Temperature')
plt.ylabel('M')
plt.errorbar(T_range, M_mean, yerr=M_err, xerr=None)
plt.show()

plt.xlabel('Temperature')
plt.ylabel('Cv')
plt.errorbar(T_range, Cv_mean, yerr=Cv_err, xerr=None)
plt.show()

plt.xlabel('Temperature')
plt.ylabel('X')
plt.errorbar(T_range, X_mean, yerr=X_err, xerr=None)
plt.show()
