from Parameters import *
from RBM import *
import numpy as np
import concurrent.futures
import copy
import os
import matplotlib.pyplot as plt
import winsound
# The code below to train an RBM at each temperatureand save the
# RBM generated data obtained via the daydream method


for nH in nH_list[3:]:
    # nH is the number of hidden nodes of the RBMs
    nH_name = 'nH = ' + str(nH)

    load_path = os.path.join('Data', 'Training Data')
    save_data_path = os.path.join('Data', 'RBM Generated Data', nH_name)
    save_weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
    save_error_path = os.path.join('Data', 'RBM Parameters', 'Errors', nH_name)

    def train_and_sample(T):
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeLoad = os.path.join(load_path, file_name)
        samples = (np.load(completeLoad) + 1)/2 # convert to 0, 1
        sz, N, N1 = samples.shape
        samples_flat = np.reshape(samples, (sz, N * N1))
        r = RBM(num_visible = 64, num_hidden = nH)
        r.train(samples_flat, max_epochs = me, learning_rate = lr)
        print("Wights at T = " + format(T, '.2f') + ": ", r.weights)
        RBM_data_flat = r.daydream(ns) * 2 - 1 # convert back to -1, 1
        RBM_data = np.reshape(RBM_data_flat, (ns, N, N1))
        completeSaveData = os.path.join(save_data_path, file_name)
        np.save(completeSaveData, RBM_data)
        completeSaveWeight = os.path.join(save_weight_path, file_name)
        np.save(completeSaveWeight, r.weights)
        completeSaveError = os.path.join(save_error_path, file_name)
        np.save(completeSaveError, r.errors)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(train_and_sample, T_range)
    fig, ax = plt.subplots()
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeSaveError = os.path.join(save_error_path, file_name)
        completeSaveError_test = os.path.join(save_error_path, file_name)
        errs = np.load(completeSaveError)
        errs_test = np.load(completeSaveError_test)
        ax.plot(errs, label = 'T = ' + format(T, '.2f'))

    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title('Gradient Descent with ' + nH_name + ' and learning rate = ' + str(lr))
    plot_name = os.path.join('Plots', 'RBM Training', nH_name + ', learning rate = ' + str(lr) + '.jpg')
    fig.savefig(plot_name, bbox_inches='tight', dpi = 1200)

    winsound.Beep(440,1000)
