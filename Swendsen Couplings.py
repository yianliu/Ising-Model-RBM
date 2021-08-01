from Parameters import *
from MyFonts import *
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import time

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

    def check(self, spins):
        (m, n) = spins.shape
        coup_from_ind = 0
        for la in range(m):
            for lb in range(n):
                l = [la, lb]
                coup_from_ind += spins[la, lb] * self.S(spins, l)
        coup_from_ind = coup_from_ind / self.order
        coup_from_all = self.all_sites_operator(spins)
        print('Computed from Single-Spin Operator:', coup_from_ind)
        print('Computed from All-Spin Operator:', coup_from_all)



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
K_1 = Coupling(name = 'Nearest Neighbour', sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2, all_sites_operator = K_1_all_sites)


# nearest neighbour coupling for the boundary spins

def K_1_bound_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m):
        val += spins[la, 0] * spins[la, n - 1]
    for lb in range(n):
        val += spins[0, lb] * spins[m - 1, lb]
    return val
K_1_bound = Coupling(name = 'Nearest Neighbour Boundary', sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2, bound_cond = True, all_sites_operator = K_1_bound_all_sites)

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
K_2 = Coupling(name = 'Next Nearest Neighbour', sites = ((-1, -1), (-1, 1), (1, -1), (1, 1)), order = 2, all_sites_operator = K_2_all_sites)

# coupling between spins that are 2 sites apart (K4 in CHung & Kao)
def K_4_all_sites(spins):
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
K_4 = Coupling(name = 'K4', sites = ((-2, 0), (2, 0), (0, -2), (0, 2)), order = 2, all_sites_operator = K_4_all_sites)

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
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    nH_name = 'nH = ' + str(nH)
    data_path = os.path.join('Data', 'RBM Generated Data', nH_name, file_name)
    samples = np.load(data_path)
    np.random.shuffle(samples)
    data_bs_sets = np.split(samples, bs_n)
    K = np.tile(np.asarray([i[1] for i in Ham]), (bs_n, 1))
    K_lst = []
    for itr in range(num_itr):
        print('Iteration', itr + 1)
        for bs_ind, dataset in enumerate(data_bs_sets):
            K_loc = K[bs_ind, :]
            for i in range(len(Ham)):
                Ham[i][1] = K_loc[i]
            Jac, S_diff = Jac_and_diff(Ham, dataset)
            h = np.linalg.solve(Jac, - S_diff)
            K_loc += h
            K[bs_ind, :] = K_loc
            print('Dataset', bs_ind + 1, 'finished' + ', time = ' + format(time.time() - start, '.2f'))
            print('Couplings:', K_loc)
        K_lst.append(copy.deepcopy(K))
        # print('Iteration', itr + 1, K)
    K_means_errs = np.stack((np.apply_along_axis(np.mean, 0, K_lst[-1]), np.apply_along_axis(error, 0, K_lst[-1])))
    return K_means_errs, K_lst

def Newton_Raphdson_MCMC(T, bs_n, Ham, num_itr):
    start = time.time()
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    data_path = os.path.join('Data', 'Training Data', file_name)
    samples = np.load(data_path)
    np.random.shuffle(samples)
    data_bs_sets = np.split(samples, bs_n)
    K = np.tile(np.asarray([i[1] for i in Ham]), (bs_n, 1))
    K_lst = []
    for itr in range(num_itr):
        print('Iteration', itr + 1)
        for bs_ind, dataset in enumerate(data_bs_sets):
            K_loc = K[bs_ind, :]
            for i in range(len(Ham)):
                Ham[i][1] = K_loc[i]
            Jac, S_diff = Jac_and_diff(Ham, dataset)
            h = np.linalg.solve(Jac, - S_diff)
            K_loc += h
            K[bs_ind, :] = K_loc
            print('Dataset', bs_ind + 1, 'finished' + ', time = ' + format(time.time() - start, '.2f'))
            print(K_loc)
        K_lst.append(copy.deepcopy(K))
        # print('Iteration', itr + 1, K)
    K_means_errs = np.stack((np.apply_along_axis(np.mean, 0, K_lst[-1]), np.apply_along_axis(error, 0, K_lst[-1])))
    return K_means_errs, K_lst

# H_1 = [[K_1, 0.4], [K_1_bound, 0.1], [K_2, 0.1]]
# nH = 64
# nH_name = 'nH = ' + str(nH)
# coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, 'H_1')
#
# for T in T_range:
#     file_name = 'T = ' + format(T, '.2f') + '.npy'
#     lst_path = os.path.join(coup_path, 'K List', file_name)
#     vals_path = os.path.join(coup_path, 'K Means and Errors', file_name)
#     K_means_errs, K_lst = Newton_Raphdson(T, nH, bs_n = 10, Ham = H_1, num_itr = 4)
#     np.save(lst_path, K_lst)
#     np.save(vals_path, K_means_errs)
#     print('T = ' + format(T, '.2f') + ':', K_means_errs)

# for T in T_range:
#     file_name = 'T = ' + format(T, '.2f') + '.npy'
#     vals_path = os.path.join(coup_path, 'K Means and Errors', file_name)
#     K_means_errs = np.load(vals_path)
#     print('\nT = ' + format(T, '.2f'))
#     for i, [val, err] in enumerate(K_means_errs.T):
#         print('K' + str(i + 1) + ' = ' + format(val, '.5f'), u"\u00B1", format(err, '.5f'))

H_2 = [[K_1, 0.4], [K_2, 0.1], [K_4, 0.1]]
nH = 64
nH_name = 'nH = ' + str(nH)
coup_path = os.path.join('Data', 'RBM Parameters', 'Couplings Swendsen', nH_name, 'H_2')

for T in T_range:
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    lst_path = os.path.join(coup_path, 'K List', file_name)
    vals_path = os.path.join(coup_path, 'K Means and Errors', file_name)
    K_means_errs, K_lst = Newton_Raphdson(T, nH, bs_n = 10, Ham = H_2, num_itr = 4)
    np.save(lst_path, K_lst)
    np.save(vals_path, K_means_errs)
    print('T = ' + format(T, '.2f') + ':', K_means_errs)
