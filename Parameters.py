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
# by each RBM; nH_list: list of numbers of hidden nodes) for RBM training and sampling
me = 5000
lr = 0.02
ns = 20000
nH_list = [16, 32, 64]

continued_me = 1000
continued_lr = 0.005
