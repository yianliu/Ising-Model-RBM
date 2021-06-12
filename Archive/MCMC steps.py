, # -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:05:31 2021

@author: Yian Liu
"""

import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from ThermoFunctions import *

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

# eqsteps is a function which takes n (dim of lattice), T (temperature),
# attempts (number of times the eqsteps is calculated) and err (the allowed
# difference between two Markov chains for which we say they have converges)
# and returns the number of steps it takes to reach equilibrium
def eqsteps(n, T, attempts, err):
    step = 0
    for i in range(attempts):
        spins_1 = rdspins(n)
        spins_2 = np.ones((n, n))
        M1 = Mag(spins_1)
        M2 = Mag(spins_2)
        step_loc = 0
        if M1 == 0:
            gap = 1
        else:
            gap = abs((M1 - M2) / M1)
        while gap >= err:
            spins = MH(spins_1, T, 1)
            spins_2 = MH(spins_2, T, 1)
            M1 = Mag(spins)
            M2 = Mag(spins_2)
            step_loc += 1
            if M1 == 0:
                gap = 1
            else:
                gap = abs((M1 - M2) / M1)
        if step < step_loc:
            step = step_loc
    return step

# Below is the temperature range for which the eqsteps will be calculated
# The temperat range is narrower than the MCMC range since the gap does not
# converge at low temperatures
nt = 10
T_range = np.linspace(2, 3, nt)

# eqsteps_lst is similar to eqsteps but takes a list of temperatures and
# computes the convergence rate for each temperature value
def eqsteps_lst(n, T_lst, attempts, err):
    for T in T_lst:
        print('T = ' + format(T, '.2f') + ': eqsteps = ', eqsteps(n, T, attempts, err))

eqsteps_lst(8, T_range, 30, 0.01)
