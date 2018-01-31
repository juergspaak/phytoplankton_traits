"""@author: J.W: Spaak

Script to show that omitting back ground absorption does not change results
"""

import numpy as np
import analytical_communities as com
from real_I_out import real_I_out_r_i

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.integrate import odeint
n = 10000
species = com.gen_species(n)
k,H,p,l = species

# I_in over time
# incoming light is sinus shaped
period = 10
size = 40
I = lambda t: size*np.sin(t/period*2*np.pi)+125 #sinus
I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period

backgrounds = np.linspace(0,10,11)
backgrounds[0] = 0.001
inv_gr = np.empty((len(backgrounds), n))

for i, background in enumerate(backgrounds):
    print(background)
    # compute the invasions growth rates for different backgrounds
    inv_gr[i] =np.amin(real_I_out_r_i(species, I, period, k_back = background),
                        axis = 0)/period


data = pd.DataFrame(inv_gr.T)
sns.violinplot(data)

plt.figure()
plt.plot(backgrounds, np.sum(inv_gr>0,axis = 1)/n,'o')
"""
pos = np.logical_and(inv_gr[-1]>0, inv_gr[0]<0)
ind = np.random.randint(sum(pos))
species = species[...,pos]
k,H,p,l = species[...,ind]

k_back = backgrounds[-1]
def dwdt(W,t, k_back):
    W_star = p/(k*l)*np.log((H+I(t))/(H+I(t)*np.exp(-k*W+k_back)))
    return ((k*W)/(k_back+sum(k*W))*W_star-W)*l
equi = com.equilibrium(species,125)
sols0 = odeint(dwdt, equi[:,ind], np.linspace(0,100*period,1000), args = (backgrounds[-1],))
sols1 = odeint(dwdt, equi[:,ind], np.linspace(0,100*period,1000), args = (0,))

plt.figure()
plt.plot(sols0)
plt.figure()
plt.plot(sols1)"""