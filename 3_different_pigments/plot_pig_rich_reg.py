# -*- coding: utf-8 -*-
"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Plots the regression of pigment richness, real data, purly random and 
model prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
species_id = "eawag-data"
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
elif species_id == "eawag-data":
    pig_spe_id = np.genfromtxt("pig_spe_id.csv",
                               delimiter = ";")[1:-1,1:-1]
    
    # pigments at 3,6,8 are for structurizing only
    pig_spe_id = pig_spe_id[np.array([0,1,2,4,5,7,8]+list(range(10,23)))]
    pig_spe_id = pig_spe_id >0.1
    r_pig, r_spe = pig_spe_id.shape
elif species_id == "paper-data":
    pig_spe_id = np.genfromtxt("")
    
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

richnesses = np.array([np.sum(r_pig_exp==r+1, axis =1) for r in 
                        range(r_pig)])
richnesses = richnesses/np.sum(richnesses, axis = 0)
for i in range(richnesses.shape[-1]):
    plt.plot(np.arange(1,r_pig+1),richnesses[:,i], label = i)
    
# compute linear regression model
trans = lambda x:np.log(x)
log_r_range = trans(r_range)
av_spe = np.average(log_r_range) #average of species richness
var_spe = np.average((log_r_range-av_spe)**2 ) #variane in sp. richness


# compute the averages of pigment richness for each species richness
av_pig = np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis])/len(r_range)
var_pig = np.sum(richnesses*(np.arange(1,r_pig+1)[:,np.newaxis]-av_pig)**2\
                /np.sum(richnesses))
# compute the covariance of species rich. and pigment rich.
cov = richnesses*(np.arange(1,r_pig+1)[:,np.newaxis]-av_pig)*\
                (log_r_range-av_spe)/np.sum(richnesses)

beta = np.sum(cov)/var_pig
alpha = av_spe -beta*av_pig
plt.figure(figsize = (7,7))

# real data from paper
n_spe = np.array([1,1,1,2,2,2,3,3,5,5,7,7,7,10,10]) #number of species in exp.
n_pig = np.array([6,8,12,8,11,12,11,14,15,16,11,14,16,17,18]) #num. of pigments

#linear regression
slope, intercept, r, p, stderr = linregress(n_pig,trans(n_spe))
plt.plot(n_pig, trans(n_spe),  'g^', label = "Experimental")
plt.plot(range(6,20), intercept+slope*np.arange(6,20),'g')
plt.axis([ 5,19,trans(0.8), trans(20)])
plt.xlabel("Pigment richness")
plt.ylabel("Species richness")
plt.legend(loc = "upper left")

plt.savefig("Figure, Regression of pigments, real data")

# plot simulated data
plt.plot( np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis],axis = 0),
         log_r_range,'ro', label = "Random")

plt.plot(np.arange(r_pig) , alpha+beta*np.arange(r_pig),'r')

datas = pd.read_csv("../4_stomp_time/data/data_random_comp_light_all.csv")
num_data = np.array(datas[[str(i+1) for i in range(10)]])
ave_data = (np.arange(1,11)*num_data).sum(axis = -1)
datas["Species richness"] = ave_data

datas_case = datas[datas["case"]=="Const1"]
n_spe = trans(datas["Species richness"].values)
n_pig = datas["r_pig"].values
slope2, intercept2, r2, p, stderr = linregress(n_pig,n_spe)
plt.plot(np.arange(r_pig), intercept2+slope2*np.arange(r_pig),'b',label = "theoretical")

plt.plot(np.arange(r_pig), trans(np.arange(r_pig)),'c',label = "maximum")


plt.legend(loc = "lower right")
plt.savefig("Figure, pig_rich_reg.pdf")


plt.show()
print(alpha, beta)
print(intercept, slope, r)
print(intercept2, slope2,r2)
    