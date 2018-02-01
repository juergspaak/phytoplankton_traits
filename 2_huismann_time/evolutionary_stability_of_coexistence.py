"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Finds communities that do coexist and randomly changes their parameters 
    slightly (evolution). After doing so checks again whether those slightly 
    changed communities do coexist."""

import matplotlib.pyplot as plt
import numpy as np

import communities_analytical as com
import r_i_analytical_continuous as ana


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
invasion_fluct2 = np.amin(ana.continuous_r_i(species, I,period)[1], axis = 0)\
                         /period #normalize by period               
  
# choose communities that do coexist                         
pos = invasion_fluct>0

species = species[...,pos]
n_spec = species.shape[-1]

# to get as close to original community richness
rep = n//n_spec

# relative changes in species traits
rel_changes = [1e-5,1e-4, 0.001,0.01,0.02,0.05,1.00]
# to save the invasion growthrates
invasion_flucts = np.empty([len(rel_changes), rep*n_spec])
species = np.repeat(species, rep, axis = -1)

for i,eps in enumerate(rel_changes[:-1]):
    # relative cahnge in species
    change = np.random.uniform(1-eps, 1+eps, species.shape)
    invasion_flucts[i] = np.amin(ana.continuous_r_i(species*change, 
            I,period)[1], axis = 0)/period #normalize by period
    
# the case of 100% change, just compute new species
invasion_flucts[-1] =  np.amin(ana.continuous_r_i(com.gen_species(n_spec*rep), 
        I,period)[1], axis = 0)/period #normalize by period
    
# plot results
plt.semilogx(rel_changes, np.sum(invasion_flucts>0,axis = 1)/(rep*n_spec),'o')
plt.xlabel("Relative change in traits")
plt.ylabel("Proportion of communities that are still in coexistence")

# Check whether all species have neighberhoods with non coexisting species
inv_flu = invasion_flucts.reshape(len(rel_changes),n_spec,rep)>0
plt.figure()
plt.semilogx(rel_changes,np.sum(np.all(inv_flu,axis = -1),axis = 1)/n_spec,'o')
plt.xlabel("Relative change in traits")
plt.ylabel("Proportions of communities that always coexist")