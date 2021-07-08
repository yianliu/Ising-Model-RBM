from Parameters import *
import numpy as np
import os
import matplotlib.pyplot as plt

def weight_to_couplings(W, c):
    m, n = np.shape(W)
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Wi = W[i]
            Wj = W[j]
            numerator = (1 + np.exp(c + Wi + Wj)) * (1 + np.exp(c))
            denominator = (1 + np.exp(c + Wi)) * (1 + np.exp(c + Wj))
            H[i,j] = np.sum(np.log(numerator / denominator)) / 8
    return H


for nH in nH_list[:1]:
    nH_name = 'nH = ' + str(nH)
    save_plot_path = os.path.join('Plots', 'Couplings', nH_name)
    save_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
    weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
    h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
    for T in T_range[:1]:
        H_arr = np.zeros((num_snapshots, 64, 64))
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        interval_epochs = int(me / num_snapshots)
        # for snapshot in range(num_snapshots):
        #     epochs_snapshot = (snapshot + 1) * interval_epochs
        #     epoch_name = 'Epochs = ' + str(epochs_snapshot)
        #     W = np.load(os.path.join(weight_path, epoch_name, file_name))
        #     c = np.load(os.path.join(h_bias_path, epoch_name, file_name))
        #     H_arr[snapshot] = weight_to_couplings(W, c)
        # np.save(os.path.join(save_couplings_path, file_name), H_arr)
        #
        H_arr = np.load(os.path.join(save_couplings_path, file_name)
        fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize=(10,7))
        for i, ax in enumerate(fig.axes):
            snapshot = snapshot_list[i]
            epochs_snapshot = (snapshot + 1) * interval_epochs
            epoch_name = 'Epochs = ' + str(epochs_snapshot)
            H = H_arr[snapshot]
            ax.imshow(H)


nH = 64
T = 2.5
nH_name = 'nH = ' + str(nH)
save_plot_path = os.path.join('Plots', 'Couplings', nH_name)
save_couplings_path = os.path.join('Data', 'RBM Parameters', 'Couplings', nH_name)
weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
H_arr = np.zeros((num_snapshots, 64, 64))
file_name = 'T = ' + format(T, '.2f') + '.npy'
interval_epochs = int(me / num_snapshots)
# for snapshot in range(num_snapshots):
#     epochs_snapshot = (snapshot + 1) * interval_epochs
#     epoch_name = 'Epochs = ' + str(epochs_snapshot)
#     W = np.load(os.path.join(weight_path, epoch_name, file_name))
#     c = np.load(os.path.join(h_bias_path, epoch_name, file_name))
#     H_arr[snapshot] = weight_to_couplings(W, c)
# np.save(os.path.join(save_couplings_path, file_name), H_arr)
#
H_arr = np.load(os.path.join(save_couplings_path, file_name)

for i in range(6):
    snapshot = snapshot_list[i]
    epochs_snapshot = (snapshot + 1) * interval_epochs
    epoch_name = 'Epochs = ' + str(epochs_snapshot)
    H = H_arr[snapshot]
    print(H)
