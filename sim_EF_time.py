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
    
save_string = "data/data_photoprotection{}.csv".format(save)
ret_length = 5
   
# time points at which we compute the densities, time is in hours 
time = 24*np.array([0,2,5,10,15,20,50,100,200])

def equi_mono(phi, l, k_photo, k_abs, I_in):
    equi_estimate = phi/l*simps(I_in, dx = dlam)
    I_in = I_in.reshape((-1,1,1))
    intrinsic = phi*simps(k_photo*I_in, dx = dlam, axis = 0)-l
    if np.amin(intrinsic)<0:
        raise
    diff = np.ones(2)
    while (diff>1e-5).any():
        tot_abs = I_inf.zm*equi_estimate*k_abs
        equi_estimate_new = phi/l*simps(k_photo/(I_inf.zm*k_abs)*I_in*
                            (1-np.exp(-tot_abs)), dx = dlam, axis = 0)
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
    return_dict = {"species": n_com*[present_species], "sky":n_com*[sky],
                   "envi": n_com*[envi], "lux": n_com*[lux]}
    k_BG = I_inf.k_BG[envi]
    k_BG.shape = -1,1,1
    zm = I_inf.zm
    # generate species
    phi,l, k_photo, k_abs, alpha, feasible = gen_com(present_species,2, n_com,
                I_ins = np.array([lux*sun_spectrum[sky]]),k_BG = k_BG, zm = zm)
    
    if not feasible:
        return np.full((7,len(time)+1),np.nan)
    
    
    
    # for the rare case where less species have been generated than predicted
    n_com = k_abs.shape[-1]
    r_spec = len(present_species)
    # incoming light regime
    I_in = lux*sun_spectrum[sky]
    N_star_mono = equi_mono(phi,l, k_photo, k_abs, I_in)
    
    # compute equilibrium densities
    equi = rc.multispecies_equi(phi/l, k_photo, k_abs, I_in, k_BG, zm)[0]
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
        tot_abs = zm*(np.nansum(N*k_abs, axis = 1, keepdims = True) + k_BG)
        # growth part
        growth = phi*simps(k_photo/tot_abs*(1-np.exp(-tot_abs))
                           *I_in.reshape(-1,1,1),dx = dlam, axis = 0)
    
        return (N*(growth-l)).flatten() # flatten for odeint
        
    sol_ode = odeint(multi_growth, start_dens.reshape(-1), time)
    sol_ode.shape = len(time), r_spec, n_com
    # append equilibrium to sol
    dens = np.append(sol_ode, equi, axis = 0)
    
    # compute niche and fitness differences
    ND, FD = rc.NFD_phytoplankton(phi, l, k_photo, k_abs,  equi = equi[0], 
                      I_in = I_in, k_BG = k_BG, zm = zm)
    ###########################################################################
    # prepare return fucntions
    tot_abs = zm*np.sum(k_abs*dens[:,np.newaxis], axis = -2)
    tot_photo = zm*np.sum(k_photo*dens[:,np.newaxis], axis = -2)
    I_out = simps(tot_photo/tot_abs*I_in.reshape((-1,1))*(1-np.exp(-tot_abs))
            ,dx = dlam, axis = 1)
    # EF biovolume
    EF = np.sum(dens, axis = 1)
    return_dict.update({EF_cols[i]:EF[i] for i in range(len(EF))})
    
    # pigment richness
    r_pig = dens*alpha[:,np.newaxis] # shape max_pig, len(time), n_spec, n_com
    r_pig = np.sum(r_pig, axis = -2) # sum over species
    r_pig = np.sum(r_pig>0, axis = 0) # count number of species
    return_dict.update({r_pig_cols[i]:r_pig[i] for i in range(len(r_pig))})
    
    # species richness, species below 0.01% are assumed extinct
    r_spec = np.sum(dens >= 1e-4*EF[:,np.newaxis], axis = 1)
    return_dict.update({r_spec_cols[i]:r_spec[i] for i in range(len(r_spec))})
    
    # compute relative yield total, complementarity and selection effect
    RYE = 1/len(phi) # relative yield expected
    
    RYO = equi[0]/N_star_mono # relative yield observed
    delta_RY = RYO-RYE # deviation from expected relative yield
    
    print(delta_RY.shape, N_star_mono.shape)
    
    return_dict["complementarity"] = (len(phi)*np.mean(delta_RY, axis = 0)
                *np.mean(N_star_mono,axis = 0))
    return_dict["selection"] = (len(phi)*np.mean(delta_RY*N_star_mono,axis = 0)
                            - return_dict["complementarity"])
    
    return_dict["RYT"] = np.sum(RYO, axis = 0)
    
    # return ND, FD, ry and monoculture density from surviving species
    sort = np.argsort(equi[0], axis = 0)
    n_max = min(ret_length, len(phi))
    ND_sort, FD_sort, N_star_sort, RYO_sort = np.full((4, ret_length,n_com),
                                                          np.nan)
    ND_sort[:n_max] = ND[sort, np.arange(n_com)][:n_max]
    return_dict.update({ND_cols[i]: ND_sort[i] for i in range(len(ND_sort))})
    FD_sort[:n_max] = FD[sort, np.arange(n_com)][:n_max]
    return_dict.update({FD_cols[i]: FD_sort[i] for i in range(len(FD_sort))})
    RYO_sort[:n_max] = RYO[sort, np.arange(n_com)][:n_max]
    return_dict.update({RYO_cols[i]: RYO_sort[i] 
                        for i in range(len(RYO_sort))})
    N_star_sort[:n_max] = N_star_mono[sort, np.arange(n_com)][:n_max]
    return_dict.update({N_mono_cols[i]: N_star_sort[i]
                        for i in range(len(N_star_sort))})
    return_dict.update({I_out_cols[i]: I_out[i] for i in range(len(I_out))})
    return dens, return_dict
 
iters = 10000

n_com = 20
r_specs = np.random.randint(1,15,iters) # richness of species

# prepare the dataframe for saving all the data
EF_cols = ["EF_t={}".format(t) for t in time]+["EF_equi"]
EF_cols[0] = "EF_start"
I_out_cols = ["I_out, t={}".format(t) for t in time] + ["I_out, equi"]
r_pig_cols = ["r_pig_t={}".format(t) for t in time]+["r_pig_equi"]
r_pig_cols[0] = "r_pig_start"
r_spec_cols = ["r_spec_t={}".format(t) for t in time] + ["r_spec_equi"]
r_spec_cols[0] = "r_spec_start"
ryt_cols = ["RYT", "complementarity", "selection"]
RYO_cols = ["RY_{}".format(i) for i in range(ret_length)]
N_mono_cols = ["N_mono_{}".format(i) for i in range(ret_length)]
ND_cols = ["ND_{}".format(i) for i in range(ret_length)]
FD_cols = ["FD_{}".format(i) for i in range(ret_length)]

# light information
skys = iters*["direct full"]
lux = np.full(iters, 40, dtype = "float")

# environment information
environments = iters*["clear"]

columns = ["species", "sky", "lux", "envi"] + EF_cols \
        + r_pig_cols + r_spec_cols + ryt_cols + RYO_cols \
        + N_mono_cols + ND_cols + FD_cols + I_out_cols
                       
data = pd.DataFrame(None, columns = columns)

i = 0
average_over_10 = 0
start = timer()

while (timer()-start<1800 - average_over_10) and i < iters:
    present_species = np.random.choice(n_diff_spe, r_specs[i], 
                                       replace = True)
    
    dens, ret_values = find_EF(present_species,
            n_com, skys[i], lux[i], environments[i])
    
    data = data.append(pd.DataFrame(ret_values), ignore_index = True,
                       sort = False)
    
    try:
        if True:
            raise ValueError
        plt.figure()
        plt.semilogy(np.append(time,24*250), dens[...,0], 'o')
        plt.ylim([1e5,1e10])
    except (ValueError, NameError):
        pass
    i += 1
    if i == 10:
        average_over_10 = timer()-start
    print(i, "iteration", timer()-start)
    
data.to_csv(save_string)