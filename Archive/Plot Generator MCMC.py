from ThermoFunctions import *
from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt


# The code below will extract observables of the corresponding dataset
# and store them in the correspondingdirectory

Engs = [] # energy
Mags = [] # magnetisation
SH_Cv = [] # specific heat
MS_X = [] # magnetic susceptibility

for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join('Data', 'Training Data', file_name)
    samples = np.load(completeName)
    Engs.append(E(samples))
    Mags.append(M(samples))
    SH_Cv.append(Cv(samples, T))
    MS_X.append(X(samples, T))

[E_vals, E_errs] = np.transpose(Engs)
[M_vals, M_errs] = np.transpose(Mags)
[Cv_vals, Cv_errs] = np.transpose(SH_Cv)
[X_vals, X_errs] = np.transpose(MS_X)

obs_path = os.path.join('Data', 'Training Data', 'Observables')
np.save(os.path.join(obs_path, 'E_vals.npy'), E_vals)
np.save(os.path.join(obs_path, 'E_errs.npy'), E_errs)
np.save(os.path.join(obs_path, 'M_vals.npy'), M_vals)
np.save(os.path.join(obs_path, 'M_errs.npy'), M_errs)
np.save(os.path.join(obs_path, 'Cv_vals.npy'), Cv_vals)
np.save(os.path.join(obs_path, 'Cv_errs.npy'), Cv_errs)
np.save(os.path.join(obs_path, 'X_vals.npy'), X_vals)
np.save(os.path.join(obs_path, 'X_errs.npy'), X_errs)


# The following chunck of code plot individual thermal observable plots for
# each nH value and saves the figure in the corresponding directory

E_vals = np.load(os.path.join(obs_path, 'E_vals.npy'))
E_errs = np.load(os.path.join(obs_path, 'E_errs.npy'))
M_vals = np.load(os.path.join(obs_path, 'M_vals.npy'))
M_errs = np.load(os.path.join(obs_path, 'M_errs.npy'))
Cv_vals = np.load(os.path.join(obs_path, 'Cv_vals.npy'))
Cv_errs = np.load(os.path.join(obs_path, 'Cv_errs.npy'))
X_vals = np.load(os.path.join(obs_path, 'X_vals.npy'))
X_errs = np.load(os.path.join(obs_path, 'X_errs.npy'))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize=(8,7))
fig.suptitle('Thermal Observables \n \n MCMC Data', fontweight = 'bold')
ax1.errorbar(T_range, E_vals, yerr = E_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Energy')
ax1.set_title('Energy', fontdict = myfont_m)
ax1.set_xlabel('T', fontdict = myfont_s)
ax1.set_ylabel('E', fontdict = myfont_s)

ax2.errorbar(T_range, M_vals, yerr = M_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Magnetisation')
ax2.set_title('Magnetisation', fontdict = myfont_m)
ax2.set_xlabel('T', fontdict = myfont_s)
ax2.set_ylabel('M', fontdict = myfont_s)

ax3.errorbar(T_range, Cv_vals, yerr = Cv_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Specific Heat')
ax3.set_title('Specific Heat', fontdict = myfont_m)
ax3.set_xlabel('T', fontdict = myfont_s)
ax3.set_ylabel('Cv', fontdict = myfont_s)

ax4.errorbar(T_range, X_vals, yerr = X_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Magetic Susceptibility')
ax4.set_title('Magetic Susceptibility', fontdict = myfont_m)
ax4.set_xlabel('T', fontdict = myfont_s)
ax4.set_ylabel('X', fontdict = myfont_s)

plt.tight_layout(pad = 1.5)

plot_path = os.path.join('Plots', 'MCMC Data.jpg')
fig.savefig(plot_path, dpi = 1200)
