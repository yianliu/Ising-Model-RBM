from Parameters import *
from MyFonts import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class Coupling:
    def __init__(self, sites, order):
        self.sites = sites
        self.order = order

# nearest neighbour coupling
coup_nn = Coupling(sites = ((-1, 0), (1, 0), (0, -1), (0, 1)), order = 2)

# next nearest neighbour coupling
coup_next_nn = Coupling(sites = ((-1, -1), (-1, 1), (1, -1), (1, 1)), order = 2)

# coupling between spins that are 2 sites apart (K4 in CHung & Kao)
coup_2 = Coupling(sites = ((-2, 0), (2, 0), (0, -2), (0, 2)), order = 2)


# S is the specific spin-indpendent coupling at site l
# The type of coupling it calculates depends on the argument coup_lst
def S(l, spins, coup):
    val = 0
    la, lb = l
    (m, n) = spins.shape
    for i, j in coup.sites:
        la_new = la + i
        lb_new = lb + j
        if la_new in range(m) and lb_new in range(n):
            val += spins[la_new, lb_new]
    return val
