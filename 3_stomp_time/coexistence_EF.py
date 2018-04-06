import numpy as np

import pandas as pd

from scipy.integrate import simps, odeint
from timeit import default_timer as timer

import richness_computation as rc
from generate_species import gen_com, n_diff_spe
from multispecies_functions import multispecies_equi

from I_in_functions import I_in_def
from load_pigments import dlam

import sys
# getting data from jobscript 
try:                    
    save = sys.argv[1]
except IndexError:
    save = np.random.randint(100000)

save_string = "data/data_EF_time"+str(save)+".csv"
time = np.array([0,2,5,10,15,20,50,100,500])

def pigment_richness(equi, alpha):
    return np.mean(np.sum(np.sum(equi*alpha, axis = -2)>0, axis = -2),-1)

def find_EF(r_spec, present_species, n_com):
    [phi,l], k_spec, alpha, species_id = gen_com(present_species, r_spec, 
                                        2, n_com)
    # incoming light regime
    I_in = lambda t: I_in_def(40)
    
    # compute equilibrium densities
    equi = multispecies_equi(phi/l, k_spec, I_in(0))[0]

    # starting density
    start_dens = np.full(equi.shape, 1e10)/r_spec
    
    # compute densities over time
    def multi_growth(N_r,t):
        N = N_r.reshape(-1,n_com)
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = np.nansum(N*k_spec, axis = 1, keepdims = True)
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
        return (N*(growth-l)).flatten()
    sol_ode = odeint(multi_growth, start_dens.reshape(-1), time)
    sol_ode.shape = len(time), r_spec, n_com

    ###########################################################################
    # prepare return fucntions
    EF = np.mean(np.sum(sol_ode, axis = 1),axis = -1)
    EF = np.append(EF, np.mean(np.sum(equi, axis = 0)))
    
    r_pig = rc.pigment_richness(sol_ode[:,np.newaxis] >= start_dens,alpha)
    r_pig = np.append(r_pig, rc.pigment_richness(equi, alpha))
    
    r_spec = np.mean(np.sum(sol_ode >= start_dens, axis = 1), axis = -1)
    r_spec = np.append(r_spec, np.mean(np.sum(equi>0,axis = 0)))
    return EF, r_pig, r_spec

iters = 5000
n_com = 100
r_specs = np.random.randint(1,40,iters) # richness of species

EF_cols = ["EF, t={}".format(t) for t in time]+["EF, equi"]
EF_cols[0] = "EF, start"
r_pig_cols = ["r_pig, t={}".format(t) for t in time]+["r_pig, equi"]
r_pig_cols[0] = "r_pig, start"
r_spec_cols = ["r_spec, t={}".format(t) for t in time] + ["r_spec, equi"]
r_spec_cols[0] = "r_spec, start"
columns = ["species","r_spec"] + EF_cols + r_pig_cols + r_spec_cols
data = pd.DataFrame(None, columns = columns, index = range(iters))

counter = 0
start = timer()
for i in range(10):
    print(counter)
    predef_spe = min(n_diff_spe, np.random.randint(r_specs[i])+1)
    present_species = np.random.choice(n_diff_spe, predef_spe, replace = False)
    EF, r_pig,r_spec = find_EF(r_specs[i], present_species, n_com)
    data.iloc[counter] = [present_species, r_specs[i], *EF, *r_pig, *r_spec]
    counter += 1
average_over_10 = timer()-start

while (timer()-start<3600 - average_over_10) and counter < iters:
    predef_spe = min(n_diff_spe, np.random.randint(r_specs[i])+1)
    present_species = np.random.choice(n_diff_spe, predef_spe, replace = False)
    EF, r_pig,r_spec = find_EF(r_specs[i], present_species, n_com)
    data.iloc[counter] = [present_species, r_specs[i], *EF, *r_pig, *r_spec]
    counter += 1
    
data = data[0:counter]
data.to_csv(save_string)
