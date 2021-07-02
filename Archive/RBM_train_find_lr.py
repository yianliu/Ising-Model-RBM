from Parameters import *
from RBM import *
import numpy as np
import concurrent.futures
import copy
import os
import matplotlib.pyplot as plt
import winsound


# The code below to train an RBM at different learning rates at each temperature
# and plots out the gradient descent in order to find the best lr for each RBM

nH = 64
nH_name = 'nH = ' + str(nH)


load_path = os.path.join('Data', 'Training Data')
save_data_path = os.path.join('Data', 'Learning Rate Test', 'RBM Generated Data', nH_name)
save_weight_path = os.path.join('Data', 'Learning Rate test', 'RBM Parameters', 'Weights', nH_name)
save_error_path = os.path.join('Data', 'Learning Rate Test', 'RBM Parameters', 'Errors', nH_name)

def train_and_sample(T, lr):
    temp_name = 'T = ' + format(T, '.2f') + '.npy'
    file_name = 'T = ' + format(T, '.2f') + ', lr = ' + str(lr) + '.npy'
    completeLoad = os.path.join(load_path, temp_name)
    samples = (np.load(completeLoad) + 1)/2 # convert to 0, 1
    sz, N, N1 = samples.shape
    samples_flat = np.reshape(samples, (sz, N * N1))
    r = RBM(num_visible = 64, num_hidden = nH)
    r.train(samples_flat, max_epochs = me, learning_rate = lr, batch_size = bs)
    print("Wights at T = " + format(T, '.2f') + ": ", r.weights)
    RBM_data_flat = r.daydream(ns) * 2 - 1 # convert back to -1, 1
    RBM_data = np.reshape(RBM_data_flat, (ns, N, N1))
    completeSaveData = os.path.join(save_data_path, file_name)
    np.save(completeSaveData, RBM_data)
    completeSaveWeight = os.path.join(save_weight_path, file_name)
    np.save(completeSaveWeight, r.weights)
    completeSaveError = os.path.join(save_error_path, file_name)
    np.save(completeSaveError, r.errors)

for T in T_range:
    def train_and_sample_at_T(lr):
        return train_and_sample(T, lr)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(train_and_sample_at_T, lr_list)

    fig, ax = plt.subplots()

    for lr in lr_list:
        file_name = 'T = ' + format(T, '.2f') + ', lr = ' + str(lr) + '.npy'
        completeSaveError = os.path.join(save_error_path, file_name)
        errs = np.load(completeSaveError)
        ax.plot(errs, label = 'lr = ' + str(lr))

    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title('Gradient Descent with ' + nH_name + ' and T = ' + format(T, '.2f'))
    plot_name = os.path.join('Plots', 'RBM Training', nH_name, 'T = ' + format(T, '.2f') + '.jpg')
    if os.path.isfile(plot_name):
       os.remove(plot_name)
    fig.savefig(plot_name, bbox_inches='tight', dpi = 1200)
    winsound.Beep(440,1000)
