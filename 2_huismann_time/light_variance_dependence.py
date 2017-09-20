"""
@author: J.W. Spaak

Find the dependence of the boundary growth rate in dependence of the variance
in incoming light"""

import numpy as np
import numerical_communities as com
import numerical_r_i as r_i

import matplotlib.pyplot as plt
from timeit import default_timer as timer

species_num, carbon, I_r = com.gen_species(com.photoinhibition_par)

rel_I_in = np.linspace(-1,1,1000)
rel_I_in.shape = -1,1
r_is = []
for var in [0,10,20,40,80,85]:
    start = timer()
    I_r = var+np.array([125,125])
    r_is.append(r_i.bound_growth(species_num, carbon, I_r,25))

    plt.figure()
    plt.scatter(*r_is[-1], s = 1, lw = 0)
    plt.xlabel(var)
    plt.show()
    pos = r_is[-1]>0
    coex = pos[0] & pos[1]
    print(np.sum(coex)/coex.size, coex.size, timer()-start)


r_is = np.asanyarray(r_is)
