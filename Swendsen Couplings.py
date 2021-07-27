from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class Coupling:
    def __init__(self, sites, order, bound_cond = False):
        self.sites = sites
        self.order = order
        self.bound_cond = bound_cond

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
            print('boundary')
            for i, j in self.sites:
                la_new = la + i
                lb_new = lb + j
                if la_new in range(m) and lb_new in range(n):
                    continue
                else:
                    if la_new < 0:
                        la_new += m
                        print('find bottom')
                    elif la_new > m - 1:
                        la_new -= m
                        print('find top')
                    if lb_new < 0:
                        lb_new += n
                        print('find right')
                    elif lb_new > n - 1:
                        lb_new -= n
                        print('find left')
                    val += spins[la_new, lb_new]
        return val

# nearest neighbour coupling
coup_nn = Coupling(sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2)

# nearest neighbour coupling for the boundary spins
coup_nn_bound = Coupling(sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2, bound_cond = True)

# next nearest neighbour coupling
coup_next_nn = Coupling(sites = ((-1, -1), (-1, 1), (1, -1), (1, 1)), order = 2)

# coupling between spins that are 2 sites apart (K4 in CHung & Kao)
coup_2 = Coupling(sites = ((-2, 0), (2, 0), (0, -2), (0, 2)), order = 2)
