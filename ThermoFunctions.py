# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:38:30 2021

@author: Yian Liu
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import os

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
    Eng = []
    for spins in spins_multi:
        Eng.append(H(spins))
    return np.asarray(Eng)

# E takes a list of multiple spin configurations and returns a list consisting
# of the mean and the standard error of energy per spin values
def E(spins_multi):
    N = spins_multi[0].size
    Eng = E_lst(spins_multi) / N
    return [np.mean(Eng), error(Eng)]

# M_lst takes a list of spin configurations and returns a list of magnetisation
# values of those configurations
def M_lst(spins_multi):
    M = []
    for spins in spins_multi:
        M.append(abs(Mag(spins)))
    return np.asarray(M)

# M takes a list of multiple spin configurations and returns a list consisting
# of the mean and the standard error of magnetisation per spin values
def M(spins_multi):
    N = spins_multi[0].size
    Mags = M_lst(spins_multi) / N
    return [np.mean(Mags), error(Mags)]

# M__real_lst takes a list of spin configurations and returns a list of magnetisation
# values of those configurations without changing the sign
def M_real_lst(spins_multi):
    M = []
    for spins in spins_multi:
        M.append(Mag(spins))
    return np.asarray(M)

# M takes a list of multiple spin configurations and returns a list consisting
# of the mean and the standard error of magnetisation per spin values without abs value
def M_real(spins_multi):
    N = spins_multi[0].size
    Mags = M_real_lst(spins_multi) / N
    return [np.mean(Mags), error(Mags)]

# Cv takes a list of multiple spin configurations and their corresponding
# temperature and returns the mean and stardard error for the specific heat
def Cv(spins_multi, T):
    N = spins_multi[0].size
    Engs = E_lst(spins_multi)
    E_mean = np.mean(Engs)
    E_err = error(Engs)
    E2 = []
    for i in Engs:
        E2.append(i * i)
    E2_mean = np.mean(E2)
    E2_err = error(E2)
    Cv_mean = (E2_mean - E_mean**2) / (N * T**2)
    Cv_err = np.sqrt(E2_err**2 + (2 * E_mean * E_err)**2) / (N * T**2)
    return [Cv_mean, Cv_err]

# X takes a list of multiple spin configurations and their corresponding
# temperature and returns the mean and stardard error for the magnetic
# susceptibility
def X(spins_multi, T):
    N = spins_multi[0].size
    Mags = M_lst(spins_multi)
    M_mean = np.mean(Mags)
    M_err = error(Mags)
    M2 = []
    for i in Mags:
        M2.append(i * i)
    M2_mean = np.mean(M2)
    M2_err = error(M2)
    X_mean = (M2_mean - M_mean**2) / (N * T)
    X_err = np.sqrt(M2_err**2 + (2 * M_mean * M_err)**2) / (N * T)
    return [X_mean, X_err]
