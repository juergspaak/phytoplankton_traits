# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:15:03 2017

@author: spaakjue
"""
import numpy as np
import matplotlib.pyplot as plt
species_id = "rando"
if species_id == "rand":
    # probability distribution of number of pigments
    p_imp = np.array([5,8,15,16,16,16,16])/16 # only important pigments
    p_ave = np.array([1,4,5,11,14,15,16])/16 # of average importance
    p_all = np.array([1,2,4,7,12,15,16])/16 # all pigments
    p = p_all
    r_pig = 18 # number of pigments in total
    r_spe = 100 # number of species
    
    # determine which species has how many pigments
    spe_id = np.random.uniform(size = r_spe)
    # number of pigments for each species
    r_pig_spe = 4 + np.sum(np.array([spe_id>=pi for pi in p]),axis = 0)
    # pigment identity in each species
    pig_spe_id = np.zeros(shape = (r_spe, r_pig))
    new = np.random.rand(r_spe, r_pig).argsort(axis = 1)
    # randomly allocate pigments in species
    for i in range(r_spe):
        pig_spe_id[i,new[i,:r_pig_spe[i]]] = 1
else:
    pig_spe_id = np.genfromtxt("pig_spe_id.csv",
                               delimiter = ";")[1:,1:]
    r_pig, r_spe = pig_spe_id.shape



    
n_exp = 10000 #number of experiments for each richness of species
max_r_exp = 9 # maximal richness in all experiments

r_range = np.arange(1, max_r_exp+1) # range of all richness of species
r_pig_exp = np.empty((len(r_range), n_exp))
for r_exp in r_range: # r_exp is species richness in each experiment
    spe_pre = np.argpartition(np.random.rand(n_exp, r_spe)
                                    ,r_exp,axis = 1)[:,:r_exp]
    new_2 = np.array([pig_spe_id[i] for i in spe_pre])
    r_pig_exp[r_exp-1] = np.sum(np.sum(new_2,axis = 1)>0,axis = 1)


richnesses = np.array([np.sum(r_pig_exp==r+1, axis =1) for r in 
                        range(r_pig)])
richnesses = richnesses/np.sum(richnesses, axis = 0)
for i in range(richnesses.shape[-1]):
    plt.plot(np.arange(1,r_pig+1),richnesses[:,i], label = i)
    
# compute linear regression model
log_r_range = np.log(r_range)
av_spe = np.average(log_r_range) #average of species richness
var_spe = np.average((log_r_range-av_spe)**2 ) #variane in sp. richness

# compute the averages of pigment richness for each species richness
p_av = np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis])/len(r_range)
# compute the covariance of species rich. and pigment rich.
cov = richnesses*(np.arange(1,r_pig+1)[:,np.newaxis]-p_av)*\
                (log_r_range-av_spe)/np.sum(richnesses)

beta = np.sum(cov)/var_spe
alpha = p_av -beta*av_spe
plt.figure()
plt.plot(log_r_range, np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis], 
                         axis = 0), 'o')
plt.plot(log_r_range, alpha+beta*log_r_range)
plt.show()
print(alpha, beta)

    
    