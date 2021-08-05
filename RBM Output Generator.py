from Parameters import *
from RBM import *
import numpy as np
import os

interval_epochs = int(me / num_snapshots)

for nH in nH_list:
    print('start')
    nH_name = 'nH = ' + str(nH)
    load_weight_path = os.path.join('Data', 'RBM Parameters', 'Weights', nH_name)
    load_v_bias_path = os.path.join('Data', 'RBM Parameters', 'Vis Biases', nH_name)
    load_h_bias_path = os.path.join('Data', 'RBM Parameters', 'Hid Biases', nH_name)
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        for snapshot in range(num_snapshots):
            epochs_snapshot = (snapshot + 1) * interval_epochs
            epochName = 'Epochs = ' + str(epochs_snapshot)
            weights = np.load(os.path.join(load_weight_path, epochName, file_name))
            v_bias = np.load(os.path.join(load_v_bias_path, epochName, file_name))
            h_bias = np.load(os.path.join(load_h_bias_path, epochName, file_name))
            r = RBM(num_visible = 64, num_hidden = nH, T = T)
            r.weights = weights
            r.v_bias = v_bias
            r.h_bias = h_bias
            RBM_data_flat = r.daydream(num_samples = ns, gibbs_steps = 1) * 2 - 1
            RBM_data = np.reshape(RBM_data_flat, (ns, n, n))
            save_data_path = os.path.join('Data', 'RBM Generated Data', nH_name, epochName)
            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)
            complete_save_data_path = os.path.join(save_data_path, file_name)
            np.save(complete_save_data_path, RBM_data)
            print(nH_name, 'T = ' + format(T, '.2f'), epochName, ': Complete!')
