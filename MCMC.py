# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:05:31 2021

@author: Yian Liu
"""

import numpy as np
import matplotlib.pyplot as plt

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
def MH(n, T, steps):
    spins = rdspins(n)
    for step in range(steps):
        for i in range(n):
            for j in range(n):
                dh = dH(spins, i, j)
                if dh <= 0:
                    spins[i, j] = - spins[i, j]
                else:
                    prob = np.exp(- dh / T)
                    if np.random.rand() < prob:
                        spins[i, j] = - spins[i, j]
    return spins

# Now we create a loop which samples sz spin configurations for a n*n lattice
# for a range of temperatures which starts at 1 and ends at 3.5 with nt intervals
# the number of iterations taken to reach the final configurations is itr
n = 6
N = n*n
sz = 100
nt = 20
itr = 100
T_range = np.linspace(1, 3.5, nt)
# Below are the functions for different thermodynamic properties
M = np.zeros(nt) # Magnetisation
E = np.zeros(nt) # Energy
Cv = np.zeros(nt) # Specific Heat
X = np.zeros(nt) # Magnetic Susceptibility

def Mag(spins):
    mag = np.sum(spins) / N
    return mag

for i in range(nt):
    t = T_range[i]
    E1 = 0
    E2 = 0
    M1 = 0
    M2 = 0
    for sp in range (sz):
        conf = MH(n, t, itr)
        h = H(conf)
        m = Mag(conf)
        E1 += h / sz
        E2 += h*h / sz
        M1 += abs(m) / sz
        M2 += m*m / sz
    E[i] = E1 / N
    M[i] = M1 / N
    Cv[i] = (E2 - E1 * E1) / (N * t * t)
    X[i] = (M2 - M1 * M1) / (N * t)

plt.figure(1)
plt.scatter(T_range, E)
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Energy", fontsize=20)

plt.figure(2)
plt.scatter(T_range, M)
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetisation", fontsize=20)

plt.figure(3)
plt.scatter(T_range, Cv)
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Specific Heat", fontsize=20)

plt.figure(4)
plt.scatter(T_range, X)
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetic Susceptibility", fontsize=20)
