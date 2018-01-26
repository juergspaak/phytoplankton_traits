import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analytical_communities as com
import analytical_r_i_continuous as ana
import seaborn as sns
from scipy.integrate import odeint
"""
n = 10000 # numer of species to compute
species = com.gen_species(n)

# incoming light is sinus shaped
period = 10
size = 40
I = lambda t: size*np.sin(t/period*2*np.pi)+125 #sinus
I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period
# invasion growth rate with fluctuating incoming light
invasion_fluct = np.amin(ana.continuous_r_i(species, I,period)[1], axis = 0)\
                         /period #normalize by period
                         
pos = invasion_fluct>0

species = species[...,pos]
n_spec = species.shape[-1]

# 1. case
equi = com.equilibrium(species, 125)

i = 0
r = 1-i
N_start = np.ones(species[0].shape)
N_start[r] = equi[r] 

time = np.linspace(0,20*period,100)

sols = np.empty((species.shape[-1], 2,100))

for j in range(n_spec):
    sols[j] = odeint(dxdt,N_start[:,j],time, args= (species[...,j],))
"""
def dxdt(N,t,pars, I_in = I):
    k,H,p,l = pars
    I_out = I_in(t)*np.exp(-(k*N).sum())
    print(I_out)
    return p*np.log((H+I_in(t))/(H+I_out))-l*N

print(dxdt(N_start[:,0],0,species[...,0]))