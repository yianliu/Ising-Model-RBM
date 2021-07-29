from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class Coupling:
    def __init__(self, sites, order, all_sites_operator, bound_cond = False):
        self.sites = sites
        self.order = order
        self.bound_cond = bound_cond
        self.all_sites_operator = all_sites_operator

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


# nearest neighbour coupling
def nn_all_sites(spins):
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

coup_nn = Coupling(sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2, all_sites_operator = nn_all_sites)


# nearest neighbour coupling for the boundary spins

def nn_bound_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m):
        val += spins[la, 0] * spins[la, n - 1]
    for lb in range(n):
        val += spins[0, lb] * spins[m - 1, lb]
    return val
coup_nn_bound = Coupling(sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2, bound_cond = True, all_sites_operator = nn_bound_all_sites)

# next nearest neighbour coupling

def next_nn_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m - 1):
        for lb in range(n - 1):
            val += spins[la, lb] * spins[la + 1, lb + 1]
    for la in range(1, m):
        for lb in range(n - 1):
            val += spins[la, lb] * spins[la - 1, lb + 1]
    return val

coup_next_nn = Coupling(sites = ((-1, -1), (-1, 1), (1, -1), (1, 1)), order = 2, all_sites_operator = next_nn_all_sites)

# coupling between spins that are 2 sites apart (K4 in CHung & Kao)
def coup_2_all_sites(spins):
    m, n = spins.shape
    val = 0
    for la in range(m - 2):
        for lb in range(n - 2):
            for i, j in ((0, 2), (2, 0)):
                la_new = la + i
                lb_new = lb + j
                val += spins[la, lb] * spins[la_new, lb_new]
    for lb in range(n - 2):
        val += spins[m - 1, lb] * spins[m - 1, lb + 2]
    for la in range(m - 2):
        val += spins[la, n - 1] * spins[la + 2, n - 1]
    return val

coup_2 = Coupling(sites = ((-2, 0), (2, 0), (0, -2), (0, 2)), order = 2, all_sites_operator = coup_2_all_sites)


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

confgs = np.load(os.path.join('Data', 'Training Data', 'T = 2.50.npy'))
H = [[coup_nn, 1], [coup_nn_bound, 0], [coup_next_nn, 0.1]]
# partial_der(coup_nn, coup_next_nn, H, confgs)

def Jacobian(Ham, spins_lst):
    num_coup = len(Ham)
    Jac = np.zeros((num_coup, num_coup))
    for a in range(num_coup):
        coup_a = Ham[a][0]
        Jac[a, :] = [partial_der(coup_a, Ham[b][0], Ham, spins_lst) for b in range(num_coup)]
    return Jac

# Jacobian(H, confgs)

def f(coup_a, Ham, spins_lst): # S tilde - S correlation function
    num_spins, m, n = spins_lst.shape
    ma = coup_a.order
    cor_fun = np.zeros((m, n))
    for la in range(m):
        for lb in range(n):
            l = [la, lb]
            val_lst = np.zeros(num_spins)
            for i, spins in enumerate(spins_lst):
                Hl = 0
                for coup, K in Ham:
                    Sl = coup.S(spins, l)
                    Hl += K * Sl
                    if coup is coup_a:
                        Sl_a = Sl
                val_lst[i] = Sl_a * np.tanh(Hl)
            cor_fun[la, lb] = np.mean(val_lst)
            print(str(l) + ': ' + str(cor_fun[la, lb]))
    S_tilde = np.sum(cor_fun) / ma
    return S_tilde
f(coup_nn, H, confgs)
