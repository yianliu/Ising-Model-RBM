from ThermoFunctions import *
import numpy as np
import os
import matplotlib.pyplot as plt

# T_range contains nt evenly spaced temperature points between 1 and 3.5
nt = 10
T_range = np.linspace(1, 3.5, nt)

data_path = 'RBM Generated Data'
Engs = [] # energy
Mags = [] # magnetisation
SH_Cv = [] # specific heat
MS_X = [] # magnetic susceptibility

for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join(data_path, file_name)
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

plt.subplot(221)
plt.errorbar(T_range, E_vals, yerr = E_errs, fmt='o', markersize=4, capsize=6)
plt.title('Energy')
plt.xlabel('T')
plt.ylabel('E')
#plt.show()

plt.subplot(222)
plt.errorbar(T_range, M_vals, yerr = M_errs, fmt='o', markersize=4, capsize=6)
plt.title('Magnetisation')
plt.xlabel('T')
plt.ylabel('M')
#plt.show()

plt.subplot(223)
plt.errorbar(T_range, Cv_vals, yerr = Cv_errs, fmt='o', markersize=4, capsize=6)
plt.title('Specific Heat')
plt.xlabel('T')
plt.ylabel('Cv')
#plt.show()

plt.subplot(224)
plt.errorbar(T_range, X_vals, yerr = X_errs, fmt='o', markersize=4, capsize=6)
plt.title('Magnetic Susceptibility')
plt.xlabel('T')
plt.ylabel('X')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.6)
plt.savefig('trained data.jpg')
