from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt

def E_eff(H, J, spins):
    spins_rbm = ((spins + 1) / 2).flatten()
    E_eff = 4 * np.dot(spins_rbm, np.dot(H, spins_rbm.T)) - np.dot(J, spins_rbm.T) + np.sum(H)
    return E_eff

def error(lst):
    return np.std(lst, ddof = 1) / np.sqrt(np.size(lst))

def E_eff_lst(H, J, spins_multi):
    spins_num = np.shape(spins_multi)[0]
    E_lst = np.zeros(spins_num)
    for i in range(spins_num):
        E_lst[i] = E_eff(H, J, spins_multi[i])
    return E_lst

def E_eff_val_err(H, J, spins_multi):
    N = spins_multi[0].size
    E_eff_norm = E_eff_lst(H, J, spins_multi) / N
    return [np.mean(E_eff_norm), error(E_eff_norm)]

for nH in nH_list:
    nH_name = 'nH = ' + str(nH)
    load_data_path = os.path.join('Data', 'RBM Generated Data', nH_name)
    load_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
    load_lt_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', 'Linear Term', nH_name)
    obs_path = os.path.join('Data', 'RBM Generated Data', nH_name, 'Observables')
    E_vals = np.load(os.path.join(obs_path, 'E_vals.npy'))
    E_errs = np.load(os.path.join(obs_path, 'E_errs.npy'))
    # E_eff_all = []
    # for T in T_range:
    #     file_name = 'T = ' + format(T, '.2f') + '.npy'
    #     H = np.load(os.path.join(load_couplings_path, file_name))[-1]
    #     J = np.load(os.path.join(load_lt_couplings_path, file_name))
    #     data = np.load(os.path.join(load_data_path, file_name))
    #     E_eff_all.append(E_eff_val_err(H, J, data))
    # [E_eff_vals, E_eff_errs] = np.transpose(E_eff_all)
    # np.save(os.path.join(obs_path, 'E_eff_vals.npy'), E_eff_vals)
    # np.save(os.path.join(obs_path, 'E_eff_errs.npy'), E_eff_errs)

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
