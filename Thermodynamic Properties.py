
import numpy as np
import matplotlib.pyplot as plt

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
