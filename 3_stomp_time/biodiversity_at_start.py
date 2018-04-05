# -*- coding: utf-8 -*-
"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Plots the regression of pigment richness, real data, purly random and 
model prediction
"""
import numpy as np
import pandas as pd

pig_spe_id = pd.read_csv("pig_spe_id.csv",delimiter = ",")

pig_spe_id = pig_spe_id.values[:,1:] >0.0
r_pig, r_spe = pig_spe_id.shape
    
n_exp = 10000 #number of experiments for each richness of species
max_r_exp = 10 # maximal richness in all experiments

r_range = np.arange(1, max_r_exp+1) # range of all richness of species
r_pig_exp = np.empty((len(r_range), n_exp)) # richness of pigments in each exp
for r_exp in r_range: # r_exp is species richness in each experiment
    # which species are present
    spe_pre = np.argpartition(np.random.rand(n_exp, r_spe)
                                    ,r_exp,axis = 1)[:,:r_exp]
    # which pigments are present in the present species
    new_2 = np.array([pig_spe_id[:,i] for i in spe_pre])
    # sum over species to get pigment presence
    r_pig_exp[r_exp-1] = np.sum(np.sum(new_2,axis = 2)>0,axis = 1)

# prob. of finding i pigments in a community of j species is richnesses[i,j]
richnesses = np.array([np.sum(r_pig_exp==r+1, axis =1) for r in 
                        range(r_pig)])
richnesses = richnesses/np.sum(richnesses, axis = 0)
pig_rich_av = np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis],axis = 0)