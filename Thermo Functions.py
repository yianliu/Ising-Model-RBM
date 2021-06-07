# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:38:30 2021

@author: Yian Liu
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

# H is a function which takes a configurationand returns the total Energy
# of the entire lattice by calculating the Hamiltonian
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

# Mag is a function which takes a configuration and returns the total magnetisation
# of the entire lattice
def Mag(spins):
    mag = np.sum(spins)
    return mag

# error is a function which calculates the standard error of a list of values
def error(lst):
    return np.std(lst, ddof = 1) / np.sqrt(np.size(lst))

# E_lst takes a list of spin configurations and returns a list of energy values
# of those configurations
def E_lst(spins_multi):
    E = []
    for spins in spins_multi:
        E.append(H(spins))
    return np.asarray(E)
#
def E_mean(spins_multi):
    return np.mean(E_lst(spins_multi))
#
def E_err(spins_multi):
    return error(E_lst(spins_multi))

def E2_lst(spins_multi):
    E = []
    for spins in spins_multi:
        E.append(H(spins)**2)
    return np.asarray(E)
#
def E2_mean(spins_multi):
    return np.mean(E_lst(spins_multi))
#
def E2_err(spins_multi):
    return error(E_lst(spins_multi))













#
def M_lst(spins_multi):
    M = []
    for spins in spins_multi:
        M.append(Mag(spins))
    return np.asarray(M)

#
def M_mean(spins_multi):
    return np.mean(M_lst(spins_multi))

#
def M_err(spins_multi):
    return error(M_lst(spins_multi))
