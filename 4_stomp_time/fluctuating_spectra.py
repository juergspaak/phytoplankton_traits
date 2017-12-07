"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Contains functions that generate coexisting communities with changing incoming
light
"""
import sys
sys.path.append("../3_different_pigments")
import load_pigments as lp
from load_pigments import lambs, dlam
import multispecies_functions as mf
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import numpy as np

def gen_com(n_com = 1000):
    r_pig, r_spec, r_pig_spec, fac = 5, 10, 3, 3
    k_spec, alpha = mf.spectrum_species(lp.real, r_pig,r_spec,n_com,r_pig_spec)
     # specific photosynthetic efficiency and loss rate
    phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    return np.array([phi,l]), k_spec, alpha
                    
def I_in_t(t, I_in1, I_in2, period):
    t_rel = (t%period)/period
    part_1 = 2*np.abs(t_rel-0.5)
    return part_1*I_in1+(1-part_1)*I_in2

par, k_spec, alpha = gen_com(100)
phi,l = par
I_ins = [mf.I_in_def(40/300, 650,50), mf.I_in_def(40/300, 450,50)]
equi = np.empty((len(I_ins),) + phi.shape)
unfixed = np.empty((len(I_ins),phi.shape[-1]))
for i,I_in in list(enumerate(I_ins)):
    equi[i], unfixed[i] = mf.multispecies_equi(phi/l, k_spec, I_in)

# consider only communities, where algorithm found equilibria
fixed = np.logical_not(np.sum(unfixed, axis = 0))

# find cases, equilibria species change
surv = equi>0 # species that survived
# XOR(present in one, present in all)
change_dom = np.logical_xor(np.sum(surv, axis = 0), np.prod(surv, axis = 0))
change_dom =change_dom.sum(axis = 0) # at least one species

# throw away uninteressting communities (i.e. not fixed, no change of dominance)
interesting = np.logical_and(change_dom, fixed)

# throw away communities, that have not been fixed/have no change in dominance
phi = phi[...,interesting]
l = l[..., interesting]
k_spec = k_spec[...,interesting]
equi = equi[..., interesting]

# remove all species that did not survive in any of the cases
dead = np.sum(surv[...,interesting], axis = 0)==0
phi[dead] = 0
l[dead] = 1 # to aboid division by 0
k_spec[:,dead] = 0
# take maximum densitiy over all lights for the starting density of the species
start_dens = np.amax(equi, axis = 0)







