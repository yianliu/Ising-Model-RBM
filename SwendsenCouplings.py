from Parameters import *
from MyFonts import *
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import time
import winsound

# ind_lst is a list of indices of the lattice sites. It is useful for vectorisation of calculations
ind_lst = (np.stack(np.indices((n, n)), axis = -1)).reshape((n * n,2))
# np.apply_along_axis(fun, 1, ind_lst)

class Coupling:
    def __init__(self, name, sites, order, all_sites_operator, bound_cond = False):
        self.name = name
        self.sites = sites
        self.order = order
        self.bound_cond = bound_cond
        self.all_sites_operator = all_sites_operator

    def __str__(self):
        return self.name

    # S is the specific spin-indpendent coupling at site l
    # The type of coupling it calculates depends on the argument coup_lst
    def S(self, spins, l):
        if self.order == 2:
            return self.S_two_spins(spins, l)
        elif self.order == 1:
            la, lb = l
            return spins[la, lb]
        else:
            return self.S_higher_order(spins, l)

    def S_two_spins(self, spins, l):
        val = 0
        la, lb = l
        (m, n) = spins.shape
        if self.bound_cond == False:
            for i, j in self.sites:
                la_new = la + i
                lb_new = lb + j
                if la_new in range(m) and lb_new in range(n):
                    val += spins[la_new, lb_new]
        elif (la in [0, m - 1] or lb in [0, n - 1]):
            # print('boundary')
            for i, j in self.sites:
                la_new = la + i
                lb_new = lb + j
                if la_new in range(m) and lb_new in range(n):
                    continue
                else:
                    if la_new < 0:
                        la_new += m
                        # print('find bottom')
                    elif la_new > m - 1:
                        la_new -= m
                        # print('find top')
                    if lb_new < 0:
                        lb_new += n
                        # print('find right')
                    elif lb_new > n - 1:
                        lb_new -= n
                        # print('find left')
                    val += spins[la_new, lb_new]
        return val

    def S_higher_order(self, spins, l):
        val = 0
        la, lb = l
        (m, n) = spins.shape
        if self.bound_cond == False:
            for locs in self.sites:
                val_loc = 1
                for i, j in locs:
                    la_new = la + i
                    lb_new = lb + j
                    if la_new in range(m) and lb_new in range(n):
                        val_loc = val_loc * spins[la_new, lb_new]
                    else:
                        val_loc = 0
                        break
                val += val_loc
            return val
        else:
            print('Higher order couplings with periodic boundary conditions not supported')

    def returnNeighbours(self, l):
        la, lb = l
        if self.bound_cond == False:
            neighbours = []
            for i, j in self.sites:
                la_new = la + i
                lb_new = lb + j
                if la_new in range(n) and lb_new in range(n):
                    neighbours.append([la_new, lb_new])
            return neighbours
        else:
            print('Neighbour finding with periodic boundary conditions not supported')

    def check(self, n):
        spins = np.random.randint(2, size=(n, n))
        print(spins)
        coup_from_ind = 0
        for la in range(n):
            for lb in range(n):
                l = [la, lb]
                coup_from_ind += spins[la, lb] * self.S(spins, l)
        coup_from_ind = coup_from_ind / self.order
        coup_from_all = self.all_sites_operator(spins)
        print('Computed from Single-Spin Operator:', coup_from_ind)
        print('Computed from All-Spin Operator:', coup_from_all)

# K_0 is the magnet field, i.e. one-spin coupling Coefficient
def K_0_all_sites(spins):
    return np.sum(spins)
K_0_site_locs = ()
K_0 = Coupling(name = 'K0', sites = K_0_site_locs, order = 1, all_sites_operator = K_0_all_sites)
# nearest neighbour coupling
def K_1_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m - 1):
        for lb in range(n - 1):
            for i, j in ((0, 1), (1, 0)):
                la_new = la + i
                lb_new = lb + j
                val += spins[la, lb] * spins[la_new, lb_new]
    for lb in range(n - 1):
        val += spins[m - 1, lb] * spins[m - 1, lb + 1]
    for la in range(m - 1):
        val += spins[la, n - 1] * spins[la + 1, n - 1]
    return val
K_1_site_locs = ((-1, 0), (1, 0), (0, -1), (0, 1))
K_1 = Coupling(name = 'K1', sites = K_1_site_locs, order = 2, all_sites_operator = K_1_all_sites)


# nearest neighbour coupling for the boundary spins

def K_1_bound_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m):
        val += spins[la, 0] * spins[la, n - 1]
    for lb in range(n):
        val += spins[0, lb] * spins[m - 1, lb]
    return val
K_1_bound = Coupling(name = 'K1$_{boundary}$', sites = K_1_site_locs, order = 2, bound_cond = True, all_sites_operator = K_1_bound_all_sites)

# next nearest neighbour coupling

def K_2_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m - 1):
        for lb in range(n - 1):
            val += spins[la, lb] * spins[la + 1, lb + 1]
    for la in range(1, m):
        for lb in range(n - 1):
            val += spins[la, lb] * spins[la - 1, lb + 1]
    return val
K_2_site_locs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
K_2 = Coupling(name = 'K2', sites = K_2_site_locs, order = 2, all_sites_operator = K_2_all_sites)

# coupling between spins that are 2 sites apart (K4 in CHung & Kao)
def K_3_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m - 2):
        for lb in range(n - 2):
            for i, j in ((0, 2), (2, 0)):
                la_new = la + i
                lb_new = lb + j
                val += spins[la, lb] * spins[la_new, lb_new]
    for lb in range(n - 2):
        val += spins[m - 2, lb] * spins[m - 2, lb + 2]
        val += spins[m - 1, lb] * spins[m - 1, lb + 2]
    for la in range(m - 2):
        val += spins[la, n - 2] * spins[la + 2, n - 2]
        val += spins[la, n - 1] * spins[la + 2, n - 1]
    return val
K_3_site_locs = ((-2, 0), (2, 0), (0, -2), (0, 2))
K_3 = Coupling(name = 'K3', sites = K_3_site_locs, order = 2, all_sites_operator = K_3_all_sites)

# 3-spin coupling (second coupling in (b) in Chung 7 Kao)
def K_4_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m - 1):
        for lb in range(n - 1):
            val += spins[la, lb] * spins[la + 1, lb] * spins[la, lb + 1]
            val += spins[la, lb] * spins[la + 1, lb + 1] * spins[la, lb + 1]
            val += spins[la, lb] * spins[la + 1, lb] * spins[la + 1, lb + 1]
            val += spins[la + 1, lb + 1] * spins[la + 1, lb] * spins[la, lb + 1]
    return val
K_4_site_locs = (((0, 1), (1, 0)), ((0, 1), (1, 1)), ((1, 0), (1, 1)),
((0, 1), (-1, 0)), ((0, 1), (-1, 1)), ((-1, 0), (-1, 1)),
((0, -1), (1, 0)), ((0, -1), (1, -1)), ((1, 0), (1, -1)),
((0, -1), (-1, 0)), ((0, -1), (-1, -1)), ((-1, 0), (-1, -1))
)
K_4 = Coupling(name = 'K4', sites = K_4_site_locs, order = 3, all_sites_operator = K_4_all_sites)
# Hamiltoians are defined as lists
def cor_fun_ind(coup_a, coup_b, Ham, spins, l):
    Hl = 0
    for coup, K in Ham:
        Sl = coup.S(spins, l)
        Hl += K * Sl
        if coup is coup_a:
            Sl_a = Sl
        if coup is coup_b:
            Sl_b = Sl
    val = Sl_a * Sl_b / np.cosh(Hl)**2
    return val

def partial_der(coup_a, coup_b, Ham, spins_lst):
    start = time.time()
    ma = coup_a.order
    num_spins, m, n = spins_lst.shape
    cor_fun = np.zeros((m, n))
    for la in range(m):
        for lb in range(n):
            l = [la, lb]
            val_lst = [cor_fun_ind(coup_a, coup_b, Ham, spins, l) for spins in spins_lst]
            cor_fun[la, lb] = np.mean(val_lst)
            print(str(l) + ', time = ' + format(time.time() - start, '.2f'))
    return np.sum(cor_fun) / ma

# confgs = np.load(os.path.join('Data', 'Training Data', 'T = 2.50.npy'))
# H = [[K_1, 0.4], [K_1_bound, 0.1], [K_2, 0.1]]
# partial_der(K_1, K_2, H, confgs)

def Jacobian(Ham, spins_lst):
    num_coup = len(Ham)
    Jac = np.zeros((num_coup, num_coup))
    for a in range(num_coup):
        coup_a = Ham[a][0]
        Jac[a, :] = [partial_der(coup_a, Ham[b][0], Ham, spins_lst) for b in range(num_coup)]
    return Jac

# Jacobian(H, confgs)

def S_diff(coup_a, Ham, spins_lst): # S tilde - S correlation function
    num_spins, m, n = spins_lst.shape
    ma = coup_a.order
    cor_fun = np.zeros((m, n))
    for la in range(m):
        for lb in range(n):
            l = [la, lb]
            val = 0
            for i, spins in enumerate(spins_lst):
                Hl = 0
                for coup, K in Ham:
                    Sl = coup.S(spins, l)
                    Hl += K * Sl
                    if coup is coup_a:
                        Sl_a = Sl
                val += Sl_a * np.tanh(Hl) / num_spins
            cor_fun[la, lb] = val
            print(str(l) + ': ' + str(cor_fun[la, lb]))
    S_tilde = np.sum(cor_fun) / ma
    S_lst = [coup_a.all_sites_operator(spins) for spins in spins_lst]
    S = np.mean(S_lst)
    return S_tilde - S
# S_diff(K_1, H, confgs)

def Jac_and_diff(Ham, spins_lst):
    # start = time.time()
    num_coup = len(Ham)
    num_spins, m, n = spins_lst.shape
    K = np.asarray([i[1] for i in Ham])
    # print(K)
    Coups = [i[0] for i in Ham]
    Jac = np.zeros((num_coup, num_coup))
    S_diff = np.zeros(num_coup)
    def S_lst(spins, l):
        return np.asarray([coup.S(spins, l) for coup in Coups])
    def Hl(S_lst_arg):
        return np.sum(K * S_lst_arg)
    for a in range(num_coup):
        coup_a = Coups[a]
        ma = coup_a.order
        def comp_cor(l):
            S_tilde_loc = 0
            S_par_lst_loc = np.zeros(num_coup)
            for spins in spins_lst:
                S_lst_res = S_lst(spins, l)
                # print(S_lst_res)
                H_l = Hl(S_lst_res)
                S_a = S_lst_res[a]
                S_tilde_loc += S_a * np.tanh(H_l) / num_spins
                for b in range(num_coup):
                    # coup_b = Coups[b]
                    S_b = S_lst_res[b]
                    S_par_lst_loc[b] += S_a * S_b / (np.cosh(H_l) ** 2 * num_spins)
            # print(str(coup_a) + str(l) + ', time = ' + format(time.time() - start, '.2f'))
            return S_tilde_loc, S_par_lst_loc
        S_tilde_l, S_par_lst_l = np.apply_along_axis(comp_cor, 1, ind_lst).T
        S_tilde = np.sum(S_tilde_l) / ma
        S_par_lst = np.sum(S_par_lst_l, axis = 0) / ma
        S_real_lst = [coup_a.all_sites_operator(spins) for spins in spins_lst]
        S = np.mean(S_real_lst)
        S_diff[a] = S_tilde - S
        Jac[a, :] = S_par_lst
    return Jac, S_diff
# Jac_and_diff(H, confgs)

def error(lst):
    return np.std(lst, ddof = 1) / np.sqrt(np.size(lst))

def Newton_Raphdson(T, nH, bs_n, Ham, num_itr):
    start = time.time()
    Ham_new = Ham.copy()
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    nH_name = 'nH = ' + str(nH)
    data_path = os.path.join('Data', 'RBM Generated Data', nH_name, file_name)
    samples = np.load(data_path)
    np.random.shuffle(samples)
    data_bs_sets = np.split(samples, bs_n)
    K = np.tile(np.asarray([i[1] for i in Ham_new]), (bs_n, 1))
    K_lst = []
    for itr in range(num_itr):
        print('Iteration', itr + 1)
        for bs_ind, dataset in enumerate(data_bs_sets):
            K_loc = K[bs_ind, :]
            for i in range(len(Ham_new)):
                Ham_new[i][1] = K_loc[i]
            Jac, S_diff = Jac_and_diff(Ham_new, dataset)
            h = np.linalg.solve(Jac, - S_diff)
            K_loc += h
            K[bs_ind, :] = K_loc
            print('Dataset', bs_ind + 1, 'finished' + ', time = ' + format(time.time() - start, '.2f'))
            print('Couplings:', K_loc)
        K_lst.append(copy.deepcopy(K))
        # print('Iteration', itr + 1, K)
    K_means_errs = np.stack((np.apply_along_axis(np.mean, 0, K_lst[-1]), np.apply_along_axis(error, 0, K_lst[-1])))
    return K_means_errs, K_lst

def Newton_Raphdson_MCMC(T, bs_n, Ham, num_itr, num_samples):
    start = time.time()
    Ham_new = Ham.copy()
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    data_path = os.path.join('Data', 'Training Data', file_name)
    samples = np.load(data_path)
    np.random.shuffle(samples)
    data_bs_sets = np.split(samples[:num_samples], bs_n)
    K = np.tile(np.asarray([i[1] for i in Ham_new]), (bs_n, 1))
    K_lst = []
    for itr in range(num_itr):
        print('Iteration', itr + 1)
        for bs_ind, dataset in enumerate(data_bs_sets):
            K_loc = K[bs_ind, :]
            for i in range(len(Ham_new)):
                Ham_new[i][1] = K_loc[i]
            Jac, S_diff = Jac_and_diff(Ham_new, dataset)
            h = np.linalg.solve(Jac, - S_diff)
            K_loc += h
            K[bs_ind, :] = K_loc
            print('Dataset', bs_ind + 1, 'finished' + ', time = ' + format(time.time() - start, '.2f'))
            print(K_loc)
        K_lst.append(copy.deepcopy(K))
        # print('Iteration', itr + 1, K)
    K_means_errs = np.stack((np.apply_along_axis(np.mean, 0, K_lst[-1]), np.apply_along_axis(error, 0, K_lst[-1])))
    return K_means_errs, K_lst

H_1 = [[K_1, 0.4], [K_1_bound, 0.1], [K_2, 0.1]]
H_2 = [[K_1, 0.4], [K_2, 0.1], [K_3, 0.1]]
H_3 = [[K_1, 0.4], [K_2, 0], [K_4, 0]]
H_4 = [[K_1, 0.4], [K_2, 0.1], [K_3, 0.1], [K_4, 0]]
H_5 = [[K_1, 0.4], [K_1_bound, 0.1], [K_2, 0.1], [K_3, 0.1]]
H_6 = [[K_1, 0.4], [K_1_bound, 0.1], [K_2, 0.1], [K_3, 0.1], [K_4, 0]]
H_7 = [[K_0, 0.1], [K_1, 0.4], [K_2, 0.1]]

def Compute_and_Save(nH, Ham, Ham_name, num_itr, bs_n):
    nH_name = 'nH = ' + str(nH)
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, Ham_name)
    lst_path = os.path.join(coup_path, 'K List')
    if not os.path.exists(lst_path):
        os.makedirs(lst_path)
    vals_path = os.path.join(coup_path, 'K Means and Errors')
    if not os.path.exists(vals_path):
        os.makedirs(vals_path)
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        K_means_errs, K_lst = Newton_Raphdson(T, nH, bs_n, Ham, num_itr)
        np.save(os.path.join(lst_path, file_name), K_lst)
        np.save(os.path.join(vals_path, file_name), K_means_errs)
        print('T = ' + format(T, '.2f') + ':', K_means_errs)

def Print_Final_Coups(nH, Ham, Ham_name):
    nH_name = 'nH = ' + str(nH)
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, Ham_name, 'K Means and Errors')
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        K_means_errs = np.load(os.path.join(coup_path, file_name))
        print('\nT = ' + format(T, '.2f'))
        for i, [val, err] in enumerate(K_means_errs.T):
            coup = Ham[i][0]
            print(coup.name + ' = ' + format(val, '.5f'), u"\u00B1", format(err, '.5f'))

def DimensionTransform_2to1(site):
    row, col = site
    site1D = row * n + col
    return site1D

def ConstructHMatrix(Ham):
    HMatrix = np.zeros((n**2, n**2))
    for coup, K in Ham:
        if coup.order == 2:
            for ind in ind_lst:
                neighbours = coup.returnNeighbours(ind)
                for neighbour in neighbours:
                    ind1D = DimensionTransform_2to1(ind)
                    neighbor1D = DimensionTransform_2to1(neighbour)
                    HMatrix[ind1D, neighbor1D] = K
        else:
            print('Only two-spin couplings supported')
    return HMatrix


def Newton_Raphdson_at_Epoch(T, nH, bs_n, Ham, num_itr, snapshot, sample_size):
    start = time.time()
    epochs_snapshot = (snapshot + 1) * interval_epochs
    epochName = 'Epochs = ' + str(epochs_snapshot)
    Ham_new = Ham.copy()
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    nH_name = 'nH = ' + str(nH)
    data_path = os.path.join('Data', 'RBM Generated Data', nH_name, epochName, file_name)
    samples = np.load(data_path)
    np.random.shuffle(samples)
    data_bs_sets = np.split(samples[:sample_size], bs_n)
    K = np.tile(np.asarray([i[1] for i in Ham_new]), (bs_n, 1))
    K_lst = []
    for itr in range(num_itr):
        print('Iteration', itr + 1)
        for bs_ind, dataset in enumerate(data_bs_sets):
            K_loc = K[bs_ind, :]
            for i in range(len(Ham_new)):
                Ham_new[i][1] = K_loc[i]
            Jac, S_diff = Jac_and_diff(Ham_new, dataset)
            h = np.linalg.solve(Jac, - S_diff)
            K_loc += h
            K[bs_ind, :] = K_loc
            print('Dataset', bs_ind + 1, 'finished' + ', time = ' + format(time.time() - start, '.2f'))
            print('Couplings:', K_loc)
        K_lst.append(copy.deepcopy(K))
        # print('Iteration', itr + 1, K)
    K_means_errs = np.stack((np.apply_along_axis(np.mean, 0, K_lst[-1]), np.apply_along_axis(error, 0, K_lst[-1])))
    return K_means_errs, K_lst

def Compute_and_Save_at_Epoch(nH, Ham, Ham_name, num_itr, bs_n, snapshot, sample_size):
    epochs_snapshot = (snapshot + 1) * interval_epochs
    epochName = 'Epochs = ' + str(epochs_snapshot)
    print(epochName)
    nH_name = 'nH = ' + str(nH)
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, Ham_name, epochName)
    lst_path = os.path.join(coup_path, 'K List')
    if not os.path.exists(lst_path):
        os.makedirs(lst_path)
    vals_path = os.path.join(coup_path, 'K Means and Errors')
    if not os.path.exists(vals_path):
        os.makedirs(vals_path)
    for T in T_range:
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        K_means_errs, K_lst = Newton_Raphdson_at_Epoch(T, nH, bs_n, Ham, num_itr, snapshot, sample_size)
        np.save(os.path.join(lst_path, file_name), K_lst)
        np.save(os.path.join(vals_path, file_name), K_means_errs)
        print('T = ' + format(T, '.2f') + ':', K_means_errs)

def Plot_Matrix(nH, Ham, Ham_name):
    nH_name = 'nH = ' + str(nH)
    n_name = 'n = ' + str(nH)
    save_plot_path = os.path.join('Plots', 'Couplings', 'Swendsen')
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, Ham_name, 'K Means and Errors')
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(10,7))
    Coup_names = ''
    for coup, K in Ham:
        Coup_names = Coup_names + coup.name + ', '
    for i, ax in enumerate(fig.axes):
        T = T_range[i]
        T_name = 'T = ' + format(T, '.2f')
        file_name = T_name + '.npy'
        K_means = np.load(os.path.join(coup_path, file_name))[0, :]
        Ham_new = Ham.copy()
        for i in range(len(Ham_new)):
            Ham_new[i][1] = K_means[i]
        HMatrix = ConstructHMatrix(Ham_new)
        im = ax.matshow(HMatrix, label = T_name, cmap = 'inferno')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(T_name)
    fig.suptitle('Swendsen Couplings for ' + n_name + '\n Couplings: ' + Coup_names[:-2], size = 18, y = 0.95)
    plt.tight_layout(pad = 3)
    fig.colorbar(im, ax = axes)
    figname = os.path.join(save_plot_path, nH_name + ', ' + Ham_name + '.jpg')
    if os.path.isfile(figname):
       os.remove(figname)
    fig.savefig(figname, dpi = 1200)


def Plot_Ham(Ham, Ham_name):
    Coups = [i[0].name for i in Ham]
    if len(Ham) == 4:
        fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (7, 6))
        legendSpacing = 1.15
    elif len(Ham) == 5:
        fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (10, 7))
        legendSpacing = 1.1
    else:
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (9, 4))
        legendSpacing = 1.1
    fig.suptitle('Coupling Coefficients from the Swendsen Method', fontweight = 'bold')
    for i, coup in enumerate(Coups):
        ax = fig.axes[i]
        if coup == 'K1':
            ax.plot(T_range, np.ones(nt), marker = '.', ls = '-', lw = 1.4, c = 'gray', label = 'Ising')
        else:
            ax.plot(T_range, np.zeros(nt), marker = '.', ls = '-', lw = 1.4, c = 'gray', label = 'Ising')
        ax.set_xlabel('Temperature', fontdict = myfont_s)
        ax.set_ylabel(coup + ' * T')
        for nH_ind, nH in enumerate(nH_list[:1]):
            coup_vals = np.zeros(nt)
            coup_errs = np.zeros(nt)
            nH_name = 'nH = ' + str(nH)
            n_name = 'n = ' + str(nH)
            coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, Ham_name, 'K Means and Errors')
            for T_ind, T in enumerate(T_range):
                file_name = 'T = ' + format(T, '.2f') + '.npy'
                K_means_errs = np.load(os.path.join(coup_path, file_name))
                coup_vals[T_ind] = K_means_errs[0, i]
                coup_errs[T_ind] = K_means_errs[1, i]
            # print(coup_errs)
            ax.errorbar(T_range, coup_vals * T_range, yerr = coup_errs * T_range, marker = '.', ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = n_name)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc ='right', bbox_to_anchor=(legendSpacing, 0.5))
    plt.tight_layout(pad = 1.5)
    plot_path = os.path.join('Plots', 'Couplings', 'Swendsen', Ham_name + '.jpg')
    if os.path.isfile(plot_path):
       os.remove(plot_path)
    fig.savefig(plot_path,  bbox_inches = 'tight', dpi = 1200)

def Plot_over_Epochs(nH, Ham, Ham_name, T_ind_lst):
    nH_name = 'nH = ' + str(nH)
    n_name = 'n = ' + str(nH)
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (9, 4))
    fig.suptitle('Changes in Coupling Coefficients over Training Epochs \n' + n_name, fontweight = 'bold')
    Coups = [i[0].name for i in Ham]
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, Ham_name)
    for coup_ind, coup in enumerate(Coups):
        ax = axes[coup_ind]
        # ax.set_title(coup)
        ax.set_xlabel('1 / Epochs')
        ax.set_ylabel(coup)
        for T_ind in T_ind_lst:
            T = T_range[T_ind]
            T_name = 'T = ' + format(T, '.2f')
            file_name = T_name + '.npy'
            epochs_inv = np.zeros(num_snapshots)
            coup_vals = np.zeros(num_snapshots)
            coup_errs = np.zeros(num_snapshots)
            for snapshot in range(num_snapshots):
                epochs_snapshot = (snapshot + 1) * interval_epochs
                epochs_inv[snapshot] = float(1 / epochs_snapshot)
                epochName = 'Epochs = ' + str(epochs_snapshot)
                vals_path = os.path.join(coup_path, epochName, 'K Means and Errors', file_name)
                K_means_errs = np.load(vals_path)
                coup_vals[snapshot] = K_means_errs[0, coup_ind]
                coup_errs[snapshot] = K_means_errs[1, coup_ind]
            ax.errorbar(epochs_inv, coup_vals, yerr = coup_errs, marker = '.', ms = 4, ls = '-', lw = 1.4, capsize = 2, ecolor = 'C7', elinewidth = 1.5, label = T_name)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc ='right', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout(pad = 1)
    plot_path = os.path.join('Plots', 'Couplings', 'Swendsen', nH_name + ', ' + Ham_name + ' over Epochs.jpg')
    if os.path.isfile(plot_path):
       os.remove(plot_path)
    fig.savefig(plot_path,  bbox_inches = 'tight', dpi = 1200)

def Compute_and_Save_MCMC(Ham, Ham_name, num_itr, bs_n, T_ind_lst, num_samples):
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', 'MCMC', Ham_name)
    lst_path = os.path.join(coup_path, 'K List')
    if not os.path.exists(lst_path):
        os.makedirs(lst_path)
    vals_path = os.path.join(coup_path, 'K Means and Errors')
    if not os.path.exists(vals_path):
        os.makedirs(vals_path)
    for T_ind in T_ind_lst:
        T = T_range[T_ind]
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        K_means_errs, K_lst = Newton_Raphdson_MCMC(T, bs_n, Ham, num_itr, num_samples)
        np.save(os.path.join(lst_path, file_name), K_lst)
        np.save(os.path.join(vals_path, file_name), K_means_errs)
        print('T = ' + format(T, '.2f') + ':', K_means_errs)

Compute_and_Save_MCMC(Ham = H_7, Ham_name = 'H_7', num_itr = 4, bs_n = 10, T_ind_lst = [7], num_samples = 20000)

def Print_Final_Coups_MCMC(Ham, Ham_name, T_ind_lst):
    coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', 'MCMC', Ham_name, 'K Means and Errors')
    for T_ind in T_ind_lst:
        T = T_range[T_ind]
        file_name = 'T = ' + format(T, '.2f') + '.npy'
        K_means_errs = np.load(os.path.join(coup_path, file_name)) * T
        print('\nT = ' + format(T, '.2f'))
        for i, [val, err] in enumerate(K_means_errs.T):
            coup = Ham[i][0]
            print(coup.name + '* T = ' + format(val, '.5f'), u"\u00B1", format(err, '.5f'))
