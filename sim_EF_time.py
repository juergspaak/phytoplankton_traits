"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Computes the EF over time for constant incoming light

generates the data data_EF*.csv that is used in plot_EF.py"""

import numpy as np
import pandas as pd
import sys

from scipy.integrate import simps, odeint
from timeit import default_timer as timer

import phytoplankton_communities.richness_computation as rc
from phytoplankton_communities.generate_species import gen_com, n_diff_spe,dlam
from phytoplankton_communities.I_in_functions import sun_spectrum
import phytoplankton_communities.I_in_functions as I_inf

# getting data from jobscript 
try:                    
    save = int(sys.argv[1])
    np.random.seed(int(save))
except IndexError:
    save = 0
    
save_string = "data/data_EF_time{}.csv".format(save)
   
# time points at which we compute the densities, time is in hours 
time = 24*np.array([0,2,5,10,15,20,50,100,200])

def equi_mono(phi, l, k_spec, I_in):
    
    
    
    equi_estimate = phi/l*simps(I_in, dx = dlam)
    I_in = I_in.reshape((-1,1,1))
    intrinsic = phi*simps(k_spec*I_in, dx = dlam, axis = 0)-l
    if np.amin(intrinsic)<0:
        raise
    diff = np.ones(2)
    while (diff>1e-5).any():
        tot_abs = I_inf.zm*equi_estimate*k_spec
        equi_estimate_new = phi/l*simps(I_in/I_inf.zm*(1-np.exp(-tot_abs)),
                                        dx = dlam, axis = 0)
        diff = np.abs(equi_estimate_new-equi_estimate)/equi_estimate
        equi_estimate = equi_estimate_new.copy()
    return equi_estimate

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
    zm = I_inf.zm
    # generate species
    phi,l, k_spec, alpha, feasible = gen_com(present_species,2, n_com,
                I_ins = np.array([lux*sun_spectrum[sky]]),k_BG = k_BG, zm = zm)
    
    if not feasible:
        return np.full((7,len(time)+1),np.nan)
    
    
    
    # for the rare case where less species have been generated than predicted
    n_com = k_spec.shape[-1]
    r_spec = len(present_species)
    # incoming light regime
    I_in = lux*sun_spectrum[sky]
    N_star_mono = equi_mono(phi,l, k_spec, I_in)
    
    # compute equilibrium densities
    equi = rc.multispecies_equi(phi/l, k_spec, I_in, k_BG, zm)[0]
    # when species can't survive equi returns nan
    equi[np.isnan(equi)] = 0
    equi.shape = 1,*equi.shape
    
    # starting density
    start_dens = np.full(phi.shape, 1e5)/r_spec
    # compute densities over time
    def multi_growth(N_r,t):
        # compute the growth rate of the species at densities N_r and time t
        
        # odeint can only work with 1-dim array, internally convert them
        N = N_r.reshape(-1,n_com)
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = zm*(np.nansum(N*k_spec, axis = 1, keepdims = True) + k_BG)
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))
                           *I_in.reshape(-1,1,1),dx = dlam, axis = 0)
    
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
    r_pig = rc.pigment_richness(np.expand_dims(dens >= 
                        1e-4*np.nansum(dens, axis = 1, keepdims = True),1),alpha)
    
    # species richness, species below 0.1% are assumed extinct
    r_spec = np.nanmean(np.sum(dens >= 1e-4*np.nansum(dens, axis = 1, 
                                    keepdims = True), axis = 1), axis = -1)
    
    # base productivity
    fitness = phi/l
    fitness_t = [np.nanmean(fitness[d>=start_dens[0]]) for d in dens]
    
    # compute relative yield total, complementarity and selection effect
    
    RYE = 1/len(phi) # relative yield expected
    RYO = dens/N_star_mono # relative yield observed
    print(RYO.shape)
    delta_RY = RYO-RYE # deviation from expected relative yield
    
    complemenratiry = np.sum(delta_RY, axis = 1)*np.sum(N_star_mono, axis = 0)
    selection = np.sum(delta_RY*N_star_mono, axis = 1)
    
    RYT = np.sum(RYO, axis = 1)
    
    return [EF_mean, EF_var, r_pig, r_spec, fitness_t, n_com, dens,
            np.median([RYT[-1], complemenratiry[-1], selection[-1]], axis = 1)]
 
iters = 1000

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
ryt_cols = ["RYT", "complementarity", "selection"]

# light information
skys = iters*["direct full"]
lux = np.full(iters, 40, dtype = "float")

# environment information
environments = iters*["clear"]

columns = ["species","r_spec", "sky", "lux", "n_com", "envi"] + EF_cols \
        + r_pig_cols + r_spec_cols + var_cols + fit_cols + ryt_cols
                       
data = pd.DataFrame(None, columns = columns, index = range(iters))

i = 0
average_over_10 = 0
start = timer()

while (timer()-start<1800 - average_over_10) and i < iters:
    present_species = np.random.choice(n_diff_spe, r_specs[i], 
                                       replace = True)
    
    EF_mean, EF_var,  r_pig, r_spec,fit,n_com_r, dens, ryt =find_EF(present_species,
            n_com, skys[i], lux[i], environments[i])
    
    data.iloc[i] = [present_species, r_specs[i], skys[i],lux[i],n_com_r,
              environments[i],*EF_mean,*r_pig, *r_spec, *EF_var,*fit, *ryt]
    
    try:
        if i>9:
            raise ValueError
        plt.figure()
        plt.semilogy(np.append(time,24*250), dens[...,0], 'o')
        plt.ylim([1e5,1e10])
    except (ValueError, NameError):
        pass
    i += 1
    if i == 10:
        average_over_10 = timer()-start
    print(i, "iteration")
    
data = data[0:i]
data.to_csv(save_string)