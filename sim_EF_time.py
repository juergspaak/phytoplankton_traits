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
    save = int(sys.argv[1])
except IndexError:
    save = np.random.randint(100000)
    
save_string = "data/data_EF_time"+str(save)+".csv"
    
time = 24*np.array([0,2,5,10,15,20,50,100])
def pigment_richness(equi, alpha):
    return np.mean(np.sum(np.sum(equi*alpha, axis = -2)>0, axis = -2),-1)

def find_EF(present_species, n_com):
    [phi,l], k_spec, alpha = gen_com(present_species,2, n_com, case = 2,
                        I_ins = np.array([I_in_def(40)]))
    
    r_spec = len(present_species)
    # incoming light regime
    I_in = lambda t: I_in_def(40)
    # compute equilibrium densities
    equi = multispecies_equi(phi/l, k_spec, I_in(0))[0]
    # when species can't survive equi returns nan
    equi[np.isnan(equi)] = 0
    equi.shape = 1,*equi.shape
    
    # starting density
    start_dens = np.full(equi.shape, 1e7)/r_spec
    
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
    
    # append equilibrium to sol
    sol_ode = np.append(sol_ode, equi, axis = 0)

    ###########################################################################
    # prepare return fucntions
    
    # EF biovolume
    EF_mean = np.nanmean(np.sum(sol_ode, axis = 1),axis = -1)
    EF_var = np.nanvar(np.sum(sol_ode, axis = 1), axis = -1)
    
    # total absorption
    tot_abs = np.exp(-np.nansum(sol_ode[:,np.newaxis]*k_spec, axis = 2))
    abs_mean = np.nanmean(simps(tot_abs, dx = dlam, axis =1)/300, axis = -1)
    abs_var = np.nanvar(simps(tot_abs, dx = dlam, axis =1)/300, axis = -1)
    # pigment richness    
    r_pig = rc.pigment_richness(sol_ode[:,np.newaxis] >= start_dens,alpha)
    # species richness
    r_spec = np.nanmean(np.sum(sol_ode >= start_dens, axis = 1), axis = -1)
    dead = np.sum(np.all(np.isnan(equi[0]), axis = 0))/n_com
    fitness = phi/l
    fitness_start = np.nanmean(fitness)
    fitness_equi = np.nanmean(fitness[equi[0]>0])
    return EF_mean, EF_var, abs_mean, abs_var, r_pig, r_spec, dead, [fitness_start, fitness_equi]


iters = 5000
n_com = 100
r_specs = np.random.randint(1,15,iters) # richness of species

EF_cols = ["EF, t={}".format(t) for t in time]+["EF, equi"]
EF_cols[0] = "EF, start"
abs_cols = ["av. abs., t={}".format(t) for t in time]+["av. abs., equi"]
abs_cols[0] = "av. abs., start"
var_cols = [col+", var" for col in EF_cols+abs_cols]
r_pig_cols = ["r_pig, t={}".format(t) for t in time]+["r_pig, equi"]
r_pig_cols[0] = "r_pig, start"
r_spec_cols = ["r_spec, t={}".format(t) for t in time] + ["r_spec, equi"]
r_spec_cols[0] = "r_spec, start"
columns = ["species","r_spec"] + EF_cols + abs_cols + r_pig_cols + \
            r_spec_cols + var_cols + ["dead","fit_start", "fit_equi"]
data = pd.DataFrame(None, columns = columns, index = range(iters))

counter = 0
average_over_10 = 0
start = timer()

while (timer()-start<1800 - average_over_10) and counter < iters:
    present_species = np.random.choice(n_diff_spe, r_specs[counter], 
                                       replace = True)
    EF_mean, EF_var, abs_mean, abs_var, r_pig, r_spec,dead,fit=\
                    find_EF(present_species, n_com)
    data.iloc[counter] = [present_species, r_specs[counter], *EF_mean,
              *abs_mean, *r_pig, *r_spec, *EF_var, *abs_var,dead,*fit]
    counter += 1
    if counter == 10:
        average_over_10 = timer()-start
    print(counter)
    
data = data[0:counter]
data.to_csv(save_string)