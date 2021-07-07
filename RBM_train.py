from Parameters import *
from RBM import *
import numpy as np
import concurrent.futures
import copy
import os
import matplotlib.pyplot as plt
import winsound

def nH_and_T_name(nH, T):
    return 'nH = ' + str(nH) + ', T = ' + format(T, '.2f')

RBMs = dict()

# The code below to train an RBM at each temperatureand save the
# RBM generated data obtained via the daydream method

for nH in nH_list:
    # nH is the number of hidden nodes of the RBMs
    nH_name = 'nH = ' + str(nH)

    load_path = os.path.join('Data', 'Training Data')
    save_data_path = os.path.join('Data', 'RBM Generated Data', nH_name)
    save_weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
    save_v_bias_path = os.path.join('Data', 'RBM Parameters', 'Vis Biases', nH_name)
    save_h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
    save_error_path = os.path.join('Data', 'RBM Parameters', 'Errors', nH_name)

    def train_and_sample(T):
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeLoad = os.path.join(load_path, file_name)
        samples = (np.load(completeLoad) + 1)/2 # convert to 0, 1
        sz, N, N1 = samples.shape
        samples_flat = np.reshape(samples, (sz, N * N1))
        r = RBM(num_visible = 64, num_hidden = nH, T = T)
        interval_epochs = int(me / num_snapshots)
        for snapshot in range(num_snapshots):
            epochs_snapshot = (snapshot + 1) * interval_epochs
            if epochs_snapshot <= 500:
                lr = 0.01
                bs = 50
                gs = 1
            elif 500 < epochs_snapshot <= 1000:
                lr = 0.001
                bs = 100
                gs = 5
            else:
                lr = 0.0001
                bs = 200
                gs = 10
            r.train(samples_flat, max_epochs = interval_epochs, learning_rate = lr, batch_size = bs, gibbs_steps = gs)
            save_weight_snapshot_path = os.path.join(save_weight_path, 'Epochs = ' + str(epochs_snapshot))
            save_v_bias_snapshot_path = os.path.join(save_v_bias_path, 'Epochs = ' + str(epochs_snapshot))
            save_h_bias_snapshot_path = os.path.join(save_h_bias_path, 'Epochs = ' + str(epochs_snapshot))
            if not os.path.exists(save_weight_snapshot_path):
                os.makedirs(save_weight_snapshot_path)
            if not os.path.exists(save_v_bias_snapshot_path):
                os.makedirs(save_v_bias_snapshot_path)
            if not os.path.exists(save_h_bias_snapshot_path):
                os.makedirs(save_h_bias_snapshot_path)
            completeSaveWeight = os.path.join(save_weight_snapshot_path, file_name)
            np.save(completeSaveWeight, r.weights)
            completeSaveVisBiases = os.path.join(save_v_bias_snapshot_path, file_name)
            np.save(completeSaveVisBiases, r.v_bias)
            completeSaveHidBiases = os.path.join(save_h_bias_snapshot_path, file_name)
            np.save(completeSaveHidBiases, r.h_bias)

        print("Wights at T = " + format(T, '.2f') + ": ", r.weights)

        RBMs[nH_and_T_name(nH, T)] = r
        RBM_data_flat = r.daydream(num_samples = ns, gibbs_steps = gs) * 2 - 1 # convert back to -1, 1
        RBM_data = np.reshape(RBM_data_flat, (ns, N, N1))

        completeSaveData = os.path.join(save_data_path, file_name)
        np.save(completeSaveData, RBM_data)
        completeSaveError = os.path.join(save_error_path, file_name)
        np.save(completeSaveError, r.errors)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(train_and_sample, T_range)
    fig, ax = plt.subplots()
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeSaveError = os.path.join(save_error_path, file_name)
        errs = np.load(completeSaveError)
        ax.plot(errs, label = 'T = ' + format(T, '.2f'))

    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title('Gradient Descent with ' + nH_name)
    plot_name = os.path.join('Plots', 'RBM Training', nH_name + '.jpg')
    if os.path.isfile(plot_name):
       os.remove(plot_name)
    fig.savefig(plot_name, bbox_inches='tight', dpi = 1200)
    winsound.Beep(440,1000)

'''
# The code below is similar to the training code above but it reads the Weights
# of RBMs that have already been trained and continues training for better results

for nH in nH_list[2:]:
    # nH is the number of hidden nodes of the RBMs
    nH_name = 'nH = ' + str(nH)
    load_data_path = os.path.join('Data', 'Training Data')
    load_weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
    load_v_bias_path = os.path.join('Data', 'RBM Parameters', 'Vis Biases', nH_name)
    load_h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
    # load_error_path = os.path.join('Data', 'RBM Parameters', 'Errors', nH_name)
    save_data_path = os.path.join('Data', 'RBM Generated Data Continued', nH_name)
    save_weight_path = os.path.join('Data', 'RBM Parameters Continued', 'Weights', nH_name)
    save_v_bias_path = os.path.join('Data', 'RBM Parameters Continued', 'Vis Biases', nH_name)
    save_h_bias_path = os.path.join('Data', 'RBM Parameters Continued', 'Hid Biases', nH_name)
    save_error_path = os.path.join('Data', 'RBM Parameters Continued', 'Errors', nH_name)

    def continued_train_and_sample(T):
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeLoad = os.path.join(load_data_path, file_name)
        completeLoadWeight = os.path.join(load_weight_path, file_name)
        samples = (np.load(completeLoad) + 1)/2 # convert to 0, 1
        sz, N, N1 = samples.shape
        samples_flat = np.reshape(samples, (sz, N * N1))
        trained_weights = np.load(completeLoadWeight)
        r = RBM(num_visible = 64, num_hidden = nH)
        r.weights = trained_weights
        r.train(samples_flat, max_epochs = me_continued, learning_rate = lr_continued, batch_size = bs_continued, gibbs_steps = gs_continued)
        print("Wights at T = " + format(T, '.2f') + ": ", r.weights)
        RBM_data_flat = r.daydream(ns) * 2 - 1 # convert back to -1, 1
        RBM_data = np.reshape(RBM_data_flat, (ns, N, N1))
        completeSaveData = os.path.join(save_data_path, file_name)
        np.save(completeSaveData, RBM_data)
        completeSaveWeight = os.path.join(save_weight_path, file_name)
        np.save(completeSaveWeight, r.weights)
        completeSaveVisBiases = os.path.join(save_v_bias_path, file_name)
        np.save(completeSaveVisBiases, r.v_bias)
        completeSaveHidBiases = os.path.join(save_h_bias_path, file_name)
        np.save(completeSaveHidBiases, r.h_bias)
        completeSaveError = os.path.join(save_error_path, file_name)
        np.save(completeSaveError, r.errors)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(continued_train_and_sample, T_range)
    fig, ax = plt.subplots()
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        completeSaveError = os.path.join(save_error_path, 'continued ' + file_name)
        errs = np.load(completeSaveError)
        ax.plot(range(me, me + continued_me), errs, label = 'T = ' + format(T, '.2f'))

    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title('Gradient Descent with ' + nH_name + ' and learning rate = ' + str(continued_lr))
    plot_name = os.path.join('Plots', 'RBM Training', nH_name + ', learning rate = ' + str(continued_lr) + ' continued.jpg')
    fig.savefig(plot_name, bbox_inches='tight', dpi = 1200)

winsound.Beep(440,1000)'''
