"""@author: J.W.Spaak

create different fluctuating incoming light functions"""

import numpy as np
from scipy.integrate import simps
from scipy.stats import linregress
import seaborn as sns

from load_pigments import lambs, dlam, real
import richness_computation as rc
from I_in_functions import fluc_continuous

iters = 1000 # number of random settings
n_com = 100 # number of communities in each setting

max_r_spec_rich = 11

# make sure, that in each community there are at more pigments than in species
r_pigs_pre = np.random.randint(1,21,2*iters) 
r_pig_specs_pre = np.random.randint(1,max_r_spec_rich,2*iters)
# richness of pigments in community and in each species
r_pigs = r_pigs_pre[r_pigs_pre>=r_pig_specs_pre][:iters]
r_pig_specs = r_pig_specs_pre[r_pigs_pre>=r_pig_specs_pre][:iters]
#r_pig_specs[:] = 1

r_specs = np.random.randint(1,40,iters) # richness of species
facs= np.random.uniform(1,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 1000
pigments = np.random.randint(0,2,iters) 
pigments = np.array(["rand", "real"])[pigments] # real/random pigments

fit_var = np.empty(iters)
fit_max = np.empty(iters)
t_const = np.array([0,0.5]) 


fit_pig = simps(real, axis = 1, dx = dlam).reshape(-1,1,1)
for i in range(iters):  
    par, k_spec, alpha = rc.gen_com(r_pigs[i], r_specs[i], r_pig_specs[i],
                                 facs[i],"real", n_com)

    fitnesses = np.sum(fit_pig* alpha, axis = 0)
    fit_var_ = np.var(fitnesses, axis = 0)
    max_diff_ = np.amax(fitnesses,axis = 0)-np.amin(fitnesses,axis = 0)
    fit_var[i], fit_max[i] = np.mean(fit_var_), np.mean(max_diff_)
    if i%100 == 99:
        print(i)

import matplotlib.pyplot as plt
import pandas as pd
fit = fit_var
rando = np.random.rand(iters)
columns = ['r_pig', 'r_spec', 'r_pig_spec','fac', "rand"]
for i,data in enumerate([r_pigs, r_specs, r_pig_specs,facs,rando]):
    plt.figure()
    plt.plot(data,fit,'.')
    # linear regression
    slope, intercept ,r,p,stder = linregress(data, fit)
    plt.plot(np.percentile(data, [1,99]), 
             intercept + slope*np.percentile(data,[1,99]), linewidth = 2.0)
    # plot polynomial
    plt.title(columns[i])
    plt.xlabel([slope, r, p])
    
fig = plt.figure()
df = pd.DataFrame({"var" : fit_var,"max": fit_max, "r_pig" : r_pigs})
sns.violinplot(data = df, y = "var", x = "r_pig", cut = 0)
#plt.axis([None,None,0,0.002])
fig.savefig("Figure, Variance in fitness.pdf")
plt.figure()
sns.violinplot(data = df, y = "max", x = "r_pig", cut = 0)
#plt.axis([None, None, 0,0.2])
fig.savefig("Figure, Max fitness differences.pdf")