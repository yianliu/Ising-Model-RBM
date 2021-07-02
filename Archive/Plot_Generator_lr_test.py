from ThermoFunctions import *
from Parameters import *
from MyFonts import *
import concurrent.futures
import numpy as np
import os
import matplotlib.pyplot as plt

nH = 64
nH_name = 'nH = ' + str(nH)
obs_path = os.path.join('Data', 'Learning Rate Test', 'RBM Generated Data', nH_name, 'Observables')

# The code below will extract observables of the corresponding dataset
# and store them in the correspondingdirectory

data_path = os.path.join('Data', 'Learning Rate Test', 'RBM Generated Data', nH_name)

training_obs_path = os.path.join('Data', 'Training Data', 'Observables')

training_E_vals = np.load(os.path.join(training_obs_path, 'E_vals.npy'))
training_M_vals = np.load(os.path.join(training_obs_path, 'M_vals.npy'))
training_Cv_vals = np.load(os.path.join(training_obs_path, 'Cv_vals.npy'))
training_X_vals = np.load(os.path.join(training_obs_path, 'X_vals.npy'))


# def lr_obs_plot(i):
for i in range(nt):
    T = T_range[i]
    T_name = 'T = ' + format(T, '.2f')
    # Engs = [] # energy
    # Mags = [] # magnetisation
    # SH_Cv = [] # specific heat
    # MS_X = [] # magnetic susceptibility
    # for lr in lr_list:
    #     file_name = 'T = ' + format(T, '.2f') + ', lr = ' + str(lr) + '.npy'
    #     completeName = os.path.join(data_path, file_name)
    #     samples = np.load(completeName)
    #     Engs.append(E(samples))
    #     Mags.append(M(samples))
    #     SH_Cv.append(Cv(samples, T))
    #     MS_X.append(X(samples, T))
    # [E_vals, E_errs] = np.transpose(Engs)
    # [M_vals, M_errs] = np.transpose(Mags)
    # [Cv_vals, Cv_errs] = np.transpose(SH_Cv)
    # [X_vals, X_errs] = np.transpose(MS_X)
    #
    # np.save(os.path.join(obs_path, T_name + ' E_vals.npy'), E_vals)
    # np.save(os.path.join(obs_path, T_name + ' E_errs.npy'), E_errs)
    # np.save(os.path.join(obs_path, T_name + ' M_vals.npy'), M_vals)
    # np.save(os.path.join(obs_path, T_name + ' M_errs.npy'), M_errs)
    # np.save(os.path.join(obs_path, T_name + ' Cv_vals.npy'), Cv_vals)
    # np.save(os.path.join(obs_path, T_name + ' Cv_errs.npy'), Cv_errs)
    # np.save(os.path.join(obs_path, T_name + ' X_vals.npy'), X_vals)
    # np.save(os.path.join(obs_path, T_name + ' X_errs.npy'), X_errs)

    # The following chunk of code plot individual thermal observable plots for
    # each nH value and saves the figure in the corresponding directory

    E_vals = np.load(os.path.join(obs_path, T_name + ' E_vals.npy'))
    E_errs = np.load(os.path.join(obs_path, T_name + ' E_errs.npy'))
    M_vals = np.load(os.path.join(obs_path, T_name + ' M_vals.npy'))
    M_errs = np.load(os.path.join(obs_path, T_name + ' M_errs.npy'))
    Cv_vals = np.load(os.path.join(obs_path, T_name + ' Cv_vals.npy'))
    Cv_errs = np.load(os.path.join(obs_path, T_name + ' Cv_errs.npy'))
    X_vals = np.load(os.path.join(obs_path, T_name + ' X_vals.npy'))
    X_errs = np.load(os.path.join(obs_path, T_name + ' X_errs.npy'))


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize=(8,7))
    fig.suptitle('Thermal Observables \n \n number of hidden nodes: ' + str(nH) + T_name, fontweight = 'bold')
    ax1.errorbar(lr_list, E_vals, yerr = E_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Energy')
    ax1.hlines(training_E_vals[i], 0, 1, color = 'grey')
    ax1.set_title('Energy', fontdict = myfont_m)
    ax1.set_xlabel('Learning Rate', fontdict = myfont_s)
    ax1.set_ylabel('E', fontdict = myfont_s)

    ax2.errorbar(lr_list, M_vals, yerr = M_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Magnetisation')
    ax2.hlines(training_M_vals[i], 0, 1, color = 'grey')
    ax2.set_title('Magnetisation', fontdict = myfont_m)
    ax2.set_xlabel('Learning Rate', fontdict = myfont_s)
    ax2.set_ylabel('M', fontdict = myfont_s)

    ax3.errorbar(lr_list, Cv_vals, yerr = Cv_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Specific Heat')
    ax3.hlines(training_Cv_vals[i], 0, 1, color = 'grey')
    ax3.set_title('Specific Heat', fontdict = myfont_m)
    ax3.set_xlabel('Learning Rate', fontdict = myfont_s)
    ax3.set_ylabel('Cv', fontdict = myfont_s)

    ax4.errorbar(lr_list, X_vals, yerr = X_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Magetic Susceptibility')
    ax4.hlines(training_X_vals[i], 0, 1, color = 'grey')
    ax4.set_title('Magetic Susceptibility', fontdict = myfont_m)
    ax4.set_xlabel('Learning Rate', fontdict = myfont_s)
    ax4.set_ylabel('X', fontdict = myfont_s)

    plt.tight_layout(pad = 1.5)

    plot_path = os.path.join('Plots', 'RBM Training', nH_name, 'Observables' + 'T = ' + format(T, '.2f') + '.jpg')
    if os.path.isfile(plot_path):
       os.remove(plot_path)
    fig.savefig(plot_path, dpi = 1200)

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     executor.map(lr_obs_plot, range(nt))
