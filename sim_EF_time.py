"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Computes the EF over time for constant incoming light

generates the data data_EF*.csv that is used in plot_EF.py"""

import numpy as np
import pandas as pd
import sys

from scipy.integrate import simps, odeint
from timeit import default_timer as timer

import richness_computation as rc
from generate_species import gen_com, n_diff_spe
from I_in_functions import sun_spectrum
import I_in_functions as I_inf
from pigments import dlam


# getting data from jobscript 
try:                    
    save = int(sys.argv[1])
    np.random.seed(int(save))
except IndexError:
    save = np.random.randint(100000)
    
save_string = "data/data_EF_time"+str(save)+".csv"
   
# time points at which we compute the densities, time is in hours 
time = 24*np.array([0,2,5,10,15,20,50,100])

def pigment_richness(dens, alpha):
    # compute the pigment richness for given densities dens
    return np.mean(np.sum(np.sum(dens*alpha, axis = -2)>0, axis = -2),-1)

def find_EF(present_species, n_com, sky, lux, envi):
    """compute the EF over time for the species
    
    Generates random species and simulates them for `time` and solves
    for equilibrium. Computes the EF for these timepoints and basefitness
    
    Input:
    present_species: list of integeters
        Id. of present species
    n_com: int
        number of species in the community
    
    `t` indicates that this parameter is computed for each timepoint in `time`   
        
    Returns:
    EF_mean: array (shape = t)
        Average ecosystem function at the different timepoints
    EF_var: equal to EF_mean, but the variance
    r_pig: similar to EF_mean, but pig richness
    r_spec: similar to EF_mena, but species richness
    fitness_t: similar to EF_mean, but average base productivity of the species
        still present at this point
    """
    k_BG = I_inf.k_BG[envi]
    k_BG.shape = -1,1,1
    zm = I_inf.zm[envi]
    # generate species
    [phi,l], k_spec, alpha, feasible = gen_com(present_species,2, n_com,
                I_ins = np.array([lux*sun_spectrum[sky]]),k_BG = k_BG, zm = zm)
    
    if not feasible:
        return np.full((5,len(time)+1),np.nan)
    # for the rare case where less species have been generated than predicted
    n_com = k_spec.shape[-1]
    r_spec = len(present_species)
    # incoming light regime
    I_in = lambda t: lux*sun_spectrum[sky]

    # compute equilibrium densities
    equi = rc.multispecies_equi(phi/l, k_spec, I_in(0), k_BG, zm)[0]
    # when species can't survive equi returns nan
    equi[np.isnan(equi)] = 0
    equi.shape = 1,*equi.shape
    
    # starting density
    start_dens = np.full(equi.shape, 1e7)/r_spec
    # compute densities over time
    def multi_growth(N_r,t):
        # compute the growth rate of the species at densities N_r and time t
        
        # odeint can only work with 1-dim array, internally convert them
        N = N_r.reshape(-1,n_com)
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = zm*(np.nansum(N*k_spec, axis = 1, keepdims = True) + k_BG)
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
    
        return (N*(growth-l)).flatten() # flatten for odeint
        
    sol_ode = odeint(multi_growth, start_dens.reshape(-1), time)
    sol_ode.shape = len(time), r_spec, n_com
    
    # append equilibrium to sol
    dens = np.append(sol_ode, equi, axis = 0)

    ###########################################################################
    # prepare return fucntions
    
    # EF biovolume
    EF_mean = np.nanmean(np.sum(dens, axis = 1),axis = -1)
    EF_var = np.nanvar(np.sum(dens, axis = 1), axis = -1)
    
    # pigment richness    
    r_pig = rc.pigment_richness(dens[:,np.newaxis] >= start_dens,alpha)
    # species richness
    r_spec = np.nanmean(np.sum(dens >= start_dens, axis = 1), axis = -1)
    
    # base productivity
    fitness = phi/l
    fitness_t = [np.nanmean(fitness[d>=start_dens[0]]) for d in dens]
    return EF_mean, EF_var, r_pig, r_spec, fitness_t
 
iters = 5000
n_com = 100
r_specs = np.random.randint(1,15,iters) # richness of species

# prepare the dataframe for saving all the data
EF_cols = ["EF, t={}".format(t) for t in time]+["EF, equi"]
EF_cols[0] = "EF, start"
var_cols = [col+", var" for col in EF_cols]
r_pig_cols = ["r_pig, t={}".format(t) for t in time]+["r_pig, equi"]
r_pig_cols[0] = "r_pig, start"
r_spec_cols = ["r_spec, t={}".format(t) for t in time] + ["r_spec, equi"]
r_spec_cols[0] = "r_spec, start"
fit_cols = ["base_prod, t={}".format(t) for t in time] + ["base_prod, equi"]
fit_cols[0] = "base_prod, start"

# light information
skys = np.array(sorted(sun_spectrum.keys()))
skys = np.random.choice(skys, iters)
lux = np.random.choice([40, 50, 100, 200, 400, 1000],iters)

# environment information
environments = np.array(sorted(I_inf.k_BG.keys()))
environments = environments[np.random.randint(len(environments), size = iters)]

columns = ["species","r_spec", "sky", "lux", "envi"] + EF_cols + r_pig_cols + \
            r_spec_cols + var_cols + fit_cols
                       
data = pd.DataFrame(None, columns = columns, index = range(iters))

i = 0
average_over_10 = 0
start = timer()

while (timer()-start<1800 - average_over_10) and i < iters:
    present_species = np.random.choice(n_diff_spe, r_specs[i], 
                                       replace = True)
    
    EF_mean, EF_var,  r_pig, r_spec,fit=find_EF(present_species, n_com, 
                        skys[i], lux[i], environments[i])
    
    data.iloc[i] = [present_species, r_specs[i], skys[i],
              lux[i],environments[i],*EF_mean,*r_pig, *r_spec, *EF_var,*fit]
              
    i += 1
    if i == 10:
        average_over_10 = timer()-start
    print(i)
    
data = data[0:i]
data.to_csv(save_string)