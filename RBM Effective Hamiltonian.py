from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time

start = time.time()

N = n * n

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def error(lst):
    return np.std(lst, ddof = 1) / np.sqrt(np.size(lst))

def E_eff_lst(W, b, c, spins_multi):
    spins_num = np.shape(spins_multi)[0]
    E_lst = np.zeros(spins_num)
    nH = len(c)
    ki_0 = np.log(1 + np.exp(c))
    sig_c = sigmoid(c)
    ki_1 = sig_c
    ki_2 = sig_c - sig_c ** 2
    coupling_0 = np.sum(ki_0) # 0th order coupling
    coupling_1_vec = b + np.dot(ki_1, W.T) # 1st order
    coupling_2_mat = np.zeros((N, N))
    for i1 in range(N):
        for i2 in range(N):
            coupling_2_mat[i1, i2] = np.sum(ki_2 * W[i1] * W[i2])
    for i in range(spins_num):
        spins = spins_multi[i].flatten()
        coupling_1 = np.sum(coupling_1_vec * spins)
        coupling_2 = np.dot(spins, np.dot(coupling_2_mat, spins.T)) / 2
        E_eff = - coupling_0 - coupling_1 - coupling_2
        # Insert E_eff here
        # print(str(i) +':' + format(time.time() - start, '.2f'))
        E_lst[i] = E_eff
    return E_lst

def E_eff_val_err(W, b, c, spins_multi, T):
    N = spins_multi[0].size
    E_eff_norm = E_eff_lst(W, b, c, spins_multi) * T / N 
    return [np.mean(E_eff_norm), error(E_eff_norm)]

for nH in nH_list:
    nH_name = 'nH = ' + str(nH)
    load_data_path = os.path.join('Data', 'RBM Generated Data', nH_name)
    weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
    v_bias_path = os.path.join('Data', 'RBM Parameters', 'Vis Biases', nH_name)
    h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
    obs_path = os.path.join('Data', 'RBM Generated Data', nH_name, 'Observables')
    E_vals = np.load(os.path.join(obs_path, 'E_vals.npy'))
    E_errs = np.load(os.path.join(obs_path, 'E_errs.npy'))
    E_eff_all = []
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        epoch_name = 'Epochs = ' + str(me)
        W = np.load(os.path.join(weight_path, epoch_name, file_name))
        b = np.load(os.path.join(v_bias_path, epoch_name, file_name))
        c = np.load(os.path.join(h_bias_path, epoch_name, file_name))
        data = np.load(os.path.join(load_data_path, file_name))
        E_eff_all.append(E_eff_val_err(W, b, c, data, T))
        print(nH_name + ', T = ' + format(T, '.2f') + ', time = ' + format(time.time() - start, '.2f'))
    [E_eff_vals, E_eff_errs] = np.transpose(E_eff_all)
    np.save(os.path.join(obs_path, 'E_eff_vals.npy'), E_eff_vals)
    np.save(os.path.join(obs_path, 'E_eff_errs.npy'), E_eff_errs)

    E_eff_vals = np.load(os.path.join(obs_path, 'E_eff_vals.npy'))
    E_eff_errs = np.load(os.path.join(obs_path, 'E_eff_errs.npy'))

    fig, ax = plt.subplots(figsize=(6,4))
    fig.suptitle('Ising Hamiltonian vs RBM Effective Hamiltonian \n for RBM Output Configurations \n number of hidden nodes: ' + str(nH), fontweight = 'bold')
    ax.errorbar(T_range, E_vals, yerr = E_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'Ising Energy')
    ax.errorbar(T_range, E_eff_vals, yerr = E_eff_errs, marker = 'o', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = 'RBM Effective Energy')
    ax.set_xlabel('T', fontdict = myfont_s)
    ax.set_ylabel('E', fontdict = myfont_s)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join('Plots', 'RBM Output', 'Effective Hamiltonian' + nH_name + '.jpg')
    if os.path.isfile(plot_path):
       os.remove(plot_path)
    fig.savefig(plot_path, bbox_inches = 'tight', dpi = 1200)
