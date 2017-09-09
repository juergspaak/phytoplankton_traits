"""
@author: Jurg W. Spaak
Computes the boundary growth for both species with analytical solution
Assumes that I_out = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import analytical_communities as com
from timeit import default_timer as timer

species = com.gen_species(10000)
bal = com.find_balance(species)

def ana_r_i(species, I):
    """ Computes the boundary growth rates of both species
    
    r_i = (k[i]/k[r]E(W*[i]/W*[r])-1)*l[i]
    where W*[i] is the equilibrium density and the average is taken over the 
    incoming light density"""
    # compute the equilibrium density of all species for all incoming lights
    W_star = com.equilibrium(species, I)
    i = [0,1]
    r = [1,0]
    k,l = species[[0,-1]]
    rel_W_star = W_star[:,i]/W_star[:,r]
    r_i = (k[i]/k[r]*np.average(rel_W_star, axis = 0)-1)*l[i]
    return r_i


    
species = com.gen_species(10000)
bal = com.find_balance(species)
# take only species with high balance incoming light

k,H,p,l = species
i,r = [[0,1],[1,0]]
fit = np.log(p[i]/l[i]/(p[r]/l[r]))
rel_I_in = np.linspace(-1,1,1000)
rel_I_in.shape = -1,1
for var in [0,10,20,50,75]:
    # randomized arounnd 125
    I_in = var*rel_I_in[:,0]+125
    r_i = ana_r_i(species, I_in)

    plt.figure()
    plt.scatter(*r_i, s = 1, c = fit[0], lw = 0)
    plt.colorbar()
    plt.xlabel(var)
    plt.show()
    pos = r_i>0
    coex = pos[0] & pos[1]
    print(np.sum(coex)/coex.size, coex.size)


species = species[..., bal>125]
fit = fit[...,bal>125]
bal = bal[bal>125]
for var in [0,10,20,50,75]:
    # optimized around balance
    I_in = var*rel_I_in+bal
    r_i = ana_r_i(species, I_in)

    plt.figure()
    plt.scatter(*r_i, s = 1, c = fit[0], lw = 0)
    plt.colorbar()
    plt.xlabel(var)
    plt.show()
    pos = r_i>0
    coex = pos[0] & pos[1]
    print(np.sum(coex)/coex.size, coex.size)
    
