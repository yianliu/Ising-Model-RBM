from ThermoFunctions import *
from Parameters import *
import numpy as np
import os
import matplotlib.pyplot as plt

nH = 64
nH_name = 'nH = ' + str(nH)
data_path = os.path.join('RBM Generated Data', nH_name)
Engs = [] # energy
Mags = [] # magnetisation
SH_Cv = [] # specific heat
MS_X = [] # magnetic susceptibility

for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join('Data', data_path, file_name)
    samples = np.load(completeName)
    Engs.append(E(samples))
    Mags.append(M(samples))
    SH_Cv.append(Cv(samples, T))
    MS_X.append(X(samples, T))

[E_vals, E_errs] = np.transpose(Engs)
[M_vals, M_errs] = np.transpose(Mags)
[Cv_vals, Cv_errs] = np.transpose(SH_Cv)
[X_vals, X_errs] = np.transpose(MS_X)

plt.figure()
plt.suptitle(nH_name)

plt.subplot(221)
plt.errorbar(T_range, E_vals, yerr = E_errs, fmt='o', markersize=4, capsize=6)
plt.title('Energy')
plt.xlabel('T')
plt.ylabel('E')

plt.subplot(222)
plt.errorbar(T_range, M_vals, yerr = M_errs, fmt='o', markersize=4, capsize=6)
plt.title('Magnetisation')
plt.xlabel('T')
plt.ylabel('M')

plt.subplot(223)
plt.errorbar(T_range, Cv_vals, yerr = Cv_errs, fmt='o', markersize=4, capsize=6)
plt.title('Specific Heat')
plt.xlabel('T')
plt.ylabel('Cv')

plt.subplot(224)
plt.errorbar(T_range, X_vals, yerr = X_errs, fmt='o', markersize=4, capsize=6)
plt.title('Magnetic Susceptibility')
plt.xlabel('T')
plt.ylabel('X')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.6)
plot_name = os.path.join('Plots', 'RBM Output', nH_name + '.jpg')
plt.savefig(plot_name, bbox_inches = 'tight')
