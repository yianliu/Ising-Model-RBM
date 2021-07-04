import numpy as np

# T_range contains nt evenly spaced temperature points between 1 and 3.5
nt = 8
T_range = np.linspace(1.5, 2.5, nt)

# Below are the parameters (n: square lattice length; sz: size of dataset at
# each temperaturevalue; eqsteps: steps taken to reach equilibrium; mcsteps:
# number of intervals between configurations) used to build a series of MCMC datasets
# where each dataset consists of sz configurations at the corresponding temperature
n = 8
sz = 50000
eqsteps = 200
mcsteps = 200


# Below are the training parameters (me: max_epochs for training which is the
# number of training steps; lr: learning rate; ns: number of samples generated
# by each RBM; nH_list: list of numbers of hidden nodes; bs: number of
# configurations in each mini batch) for RBM training and sampling
me = 500
lr = 0.01
# lr_list contains a list of lr values to be tried on the RBMs
# lr_list = [0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
ns = 20000
nH_list = [4, 16, 64]
bs = 50
gs = 1


def T_name(T):
    return'T = ' + format(T, '.2f')

lr_nH_64 = dict()
lr_nH_64[T_name(T_range[0])] = 0.8
lr_nH_64[T_name(T_range[1])] = 0.8
lr_nH_64[T_name(T_range[2])] = 0.02
lr_nH_64[T_name(T_range[3])] = 0.5
lr_nH_64[T_name(T_range[4])] = 0.5
lr_nH_64[T_name(T_range[5])] = 0.01
lr_nH_64[T_name(T_range[6])] = 0.01
lr_nH_64[T_name(T_range[7])] = 0.01
