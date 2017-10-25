# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:15:03 2017

@author: spaakjue
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
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
max_r_exp = 10 # maximal richness in all experiments

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
# real data from paper
n_spe = np.array([1,1,1,2,2,2,3,3,5,5,7,7,7,10,10]) #number of species in exp.
n_pig = np.array([6,8,12,8,11,12,11,14,15,16,11,14,16,17,18]) #num. of pigments

#linear regression
slope, intercept, r, p, stderr = linregress(np.log(n_spe), n_pig)
plt.plot(np.log(n_spe), n_pig, 'g^', label = "Experimental")
plt.plot(np.log(n_spe),intercept+slope*np.log(n_spe),'g')
plt.xlabel("Species richness (log)")
plt.ylabel("Pigment richness")
plt.axis([-0.2, 2.5, 5,19])
plt.savefig("Figure, Regression of pigments, real data")
# plot simulated data
plt.plot(log_r_range, np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis], 
                         axis = 0), 'ro', label = "Simulated")
plt.plot(log_r_range, alpha+beta*log_r_range, 'r')
plt.legend(loc = "upper left")
plt.savefig("Figure, Regression of pigments richness")


plt.show()
print(alpha, beta)
print(intercept, slope, r)
    