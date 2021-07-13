from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt

# def weight_to_couplings(W, c):
#     m, n = np.shape(W)
#     H = np.zeros((m, m))
#     for i in range(m):
#         for j in range(m):
#             Wi = W[i]
#             Wj = W[j]
#             numerator = (1 + np.exp(c + Wi + Wj)) * (1 + np.exp(c))
#             denominator = (1 + np.exp(c + Wi)) * (1 + np.exp(c + Wj))
#             H[i,j] = np.sum(np.log(numerator / denominator)) / 8
#         H[i,i] = 0
#     return H
#
#
# for nH in nH_list:
#     nH_name = 'nH = ' + str(nH)
#     save_plot_path = os.path.join('Plots', 'Couplings', nH_name)
#     save_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
#     weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
#     h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
#     for T in T_range:
#         H_arr = np.zeros((num_snapshots, 64, 64))
#         file_name = 'T = ' + format(T, '.2f') + '.npy'
#         interval_epochs = int(me / num_snapshots)
#         for snapshot in range(num_snapshots):
#             epochs_snapshot = (snapshot + 1) * interval_epochs
#             epoch_name = 'Epochs = ' + str(epochs_snapshot)
#             W = np.load(os.path.join(weight_path, epoch_name, file_name))
#             c = np.load(os.path.join(h_bias_path, epoch_name, file_name))
#             H_arr[snapshot] = weight_to_couplings(W, c)
#         np.save(os.path.join(save_couplings_path, file_name), H_arr)
#
#         H_arr = np.load(os.path.join(save_couplings_path, file_name))
#         fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize=(10,7))
#         for i, ax in enumerate(fig.axes):
#             snapshot = snapshot_list[i]
#             epochs_snapshot = (snapshot + 1) * interval_epochs
#             epoch_name = 'Epochs = ' + str(epochs_snapshot)
#             H = H_arr[snapshot]
#             im = ax.matshow(H, label = epoch_name, cmap = 'inferno')
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_title(epoch_name)
#         fig.suptitle('T = ' + format(T, '.2f'), size = 18, y = 0.95)
#         plt.tight_layout(pad = 3)
#         fig.colorbar(im, ax = axes)
#         figname = os.path.join(save_plot_path, 'T = ' + format(T, '.2f') + '.jpg')
#         if os.path.isfile(figname):
#            os.remove(figname)
#         fig.savefig(figname, dpi = 1200)
#
# # The following code plots out the histograms of the couplings
# for nH in nH_list:
#     nH_name = 'nH = ' + str(nH)
#     n_name = 'n = ' + str(nH)
#     save_plot_path = os.path.join('Plots', 'Couplings', 'Histograms')
#     save_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
#     fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(10,5))
#     for i, ax in enumerate(fig.axes):
#         T = T_range[i]
#         file_name = 'T = ' + format(T, '.2f') + '.npy'
#         interval_epochs = int(me / num_snapshots)
#         H = np.load(os.path.join(save_couplings_path, file_name))[-1].flatten()
#         ax.hist(H, bins = 50)
#         ax.set_title('T = ' + format(T, '.2f'))
#     fig.suptitle(n_name, size = 15)
#     plt.tight_layout(pad = 2)
#     figname = os.path.join(save_plot_path, nH_name + '.jpg')
#     if os.path.isfile(figname):
#        os.remove(figname)
#     fig.savefig(figname, dpi = 1200)


# The following code plots out the histograms of the couplings away from the origin
for nH in nH_list:
    nH_name = 'nH = ' + str(nH)
    n_name = 'n = ' + str(nH)
    save_plot_path = os.path.join('Plots', 'Couplings', 'Histograms')
    save_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(10,5))
    for i, ax in enumerate(fig.axes):
        T = T_range[i]
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        interval_epochs = int(me / num_snapshots)
        H = np.load(os.path.join(save_couplings_path, file_name))[-1].flatten()
        try:
            bins = np.linspace(0.1, max(H), num = 50)
            ax.hist(H, bins = bins)
        except:
            ax.hist(H, bins = 50)
        ax.set_title('T = ' + format(T, '.2f'))
    fig.suptitle(n_name, size = 15)
    plt.tight_layout(pad = 2)
    figname = os.path.join(save_plot_path, 'non-zero ' + nH_name + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig.savefig(figname, dpi = 1200)


# The following code extracts the nearest neighbour couplings from the coupling matrices
nn_list = [-8, 8, -1, 1]
vis_ind = range(64)

for nH in nH_list:
    nH_name = 'nH = ' + str(nH)
    n_name = 'n = ' + str(nH)
    save_plot_path = os.path.join('Plots', 'Couplings', 'Histograms')
    load_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
    save_nn_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', 'Nearest Neighbour', nH_name)
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(10,5))
    for i, ax in enumerate(fig.axes):
        T = T_range[i]
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        H = np.load(os.path.join(load_couplings_path, file_name))[-1]
        H_nn = []
        # H_test = np.zeros(np.shape(H))
        for spin in vis_ind:
            for nn in nn_list:
                spin_nn = spin + nn
                if spin_nn in vis_ind:
                    Coup_nn = H[spin, spin_nn]
                    H_nn.append(Coup_nn)
                    # H_test[spin, spin_nn] = 1
        np.save(os.path.join(save_nn_couplings_path, file_name), np.asarray(H_nn))
        # plt.matshow(H_test)

        ax.hist(H_nn)
        ax.set_title('T = ' + format(T, '.2f'))
    fig.suptitle(n_name, size = 15)
    plt.tight_layout(pad = 2)
    figname = os.path.join(save_plot_path, 'Nearest Neighbour ' + nH_name + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig.savefig(figname, dpi = 1200)


# The following code plots the nearest-neighbour couplings against expectation from Ising model
# (reproduce fig. 11 in Cossu et al)
