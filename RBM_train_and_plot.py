from Parameters import *
from ThermoFunctions import *
from RBM import *
from MyFonts import *
import numpy as np
import concurrent.futures
import copy
import os
import matplotlib.pyplot as plt
import winsound


def train_and_plot(nH, T, lr, me, steps, bs, gs):

    T_pos = np.where(T_range == T)[0][0]

    epoch_int = int(me/steps)
    epoch_list = []

    load_path = os.path.join('Data', 'Training Data')

    E_training = np.load(os.path.join(load_path, 'Observables', 'E_vals.npy'))[T_pos]
    M_training = np.load(os.path.join(load_path, 'Observables', 'M_vals.npy'))[T_pos]
    Cv_training = np.load(os.path.join(load_path, 'Observables', 'Cv_vals.npy'))[T_pos]
    X_training = np.load(os.path.join(load_path, 'Observables', 'X_vals.npy'))[T_pos]

    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeLoad = os.path.join(load_path, file_name)
    samples = (np.load(completeLoad) + 1)/2 # convert to 0, 1
    sz, N, N1 = samples.shape
    samples_flat = np.reshape(samples, (sz, N * N1))
    # lr = lr_nH_64['T = ' + format(T, '.2f')]
    r = RBM(num_visible = 64, num_hidden = nH)

    Engs = [] # energy
    Mags = [] # magnetisation
    SH_Cv = [] # specific heat
    MS_X = [] # magnetic susceptibility

    for step in range(steps):
        epoch_list.append((step + 1) * epoch_int)
        r.train(samples_flat, max_epochs = epoch_int, learning_rate = lr, batch_size = bs, gibbs_steps = gs)
        RBM_data_flat = r.daydream(ns) * 2 - 1 # convert back to -1, 1
        RBM_data = np.reshape(RBM_data_flat, (ns, N, N1))
        Engs.append(E(RBM_data))
        Mags.append(M(RBM_data))
        SH_Cv.append(Cv(RBM_data, T))
        MS_X.append(X(RBM_data, T))

    [E_vals, E_errs] = np.transpose(Engs)
    [M_vals, M_errs] = np.transpose(Mags)
    [Cv_vals, Cv_errs] = np.transpose(SH_Cv)
    [X_vals, X_errs] = np.transpose(MS_X)

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize=(8,7))
    fig1.suptitle('Thermal Observables \n \n nH = ' + str(nH) + ', lr = ' + str(lr) + ', bs = ' + str(bs) + ',  T = ' + format(T, '.2f') + ', gs = ' + str(gs), fontweight = 'bold')
    ax1.axhline(E_training)
    ax1.errorbar(epoch_list, E_vals, yerr = E_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Energy')
    ax1.set_title('Energy', fontdict = myfont_m)
    ax1.set_xlabel('Steps', fontdict = myfont_s)
    ax1.set_ylabel('E', fontdict = myfont_s)

    ax2.errorbar(epoch_list, M_vals, yerr = M_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Magnetisation')
    ax2.axhline(M_training)
    ax2.set_title('Magnetisation', fontdict = myfont_m)
    ax2.set_xlabel('Steps', fontdict = myfont_s)
    ax2.set_ylabel('M', fontdict = myfont_s)

    ax3.errorbar(epoch_list, Cv_vals, yerr = Cv_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Specific Heat')
    ax3.axhline(Cv_training)
    ax3.set_title('Specific Heat', fontdict = myfont_m)
    ax3.set_xlabel('Steps', fontdict = myfont_s)
    ax3.set_ylabel('Cv', fontdict = myfont_s)

    ax4.errorbar(epoch_list, X_vals, yerr = X_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Magetic Susceptibility')
    ax4.axhline(X_training)
    ax4.set_title('Magetic Susceptibility', fontdict = myfont_m)
    ax4.set_xlabel('Steps', fontdict = myfont_s)
    ax4.set_ylabel('X', fontdict = myfont_s)

    plt.tight_layout(pad = 1.5)
    figname = os.path.join('Plots', 'RBM Training', 'nH = ' + str(nH),'lr = ' + str(lr) + ', bs = ' + str(bs) + ',  T = ' + format(T, '.2f') + ', gs = ' + str(gs) + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig1.savefig(figname, dpi = 600)
    fig1.show()

    fig2, ax5 = plt.subplots()
    ax5.plot(r.errors)
    fig2.show()

    winsound.Beep(440,1000)

train_and_plot(nH = 64, T = T_range[0], lr = 0.1, me = 25, steps = 25, bs = 200, gs = 1)
