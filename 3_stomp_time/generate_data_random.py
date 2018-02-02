"""@author: J.W.Spaak

Computes the number of coexisting species for random settings
Incoming light is a sinus/step/linear fluctuation of two gaussian curves"""

import pandas as pd
import numpy as np

from richness_computation import fluctuating_richness
import I_in_functions as I_fun

import sys
sys.path.append("../3_different_pigments")
from multispecies_functions import I_in_def

from timeit import default_timer as timer
start = timer()

iters = 10000 # number of random settings
n_com = 100 # number of communities in each setting

# make sure, that in each community there are at more pigments than in species
r_pigs_pre = np.random.randint(2,21,2*iters) 
r_pig_specs_pre = np.random.randint(1,11,2*iters)
# richness of pigments in community and in each species
r_pigs = r_pigs_pre[r_pigs_pre>r_pig_specs_pre][:iters]
r_pig_specs = r_pig_specs_pre[r_pigs_pre>r_pig_specs_pre][:iters]

r_specs = np.random.randint(2,40,iters) # richness of species
facs= np.random.uniform(1,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 1000
pigments = np.random.randint(0,2,iters) 
pigments = np.array(["rand", "real"])[pigments] # real/random pigments

## Determining the light regime for each setting
# random incoming light fluctuations
luxs = np.random.uniform(30/300,100/300,(iters,2))
sigmas = 2**np.random.uniform(5,9,(iters,2)) # ragnes from 16-512
locs = np.random.uniform(400,700,(iters,2))

I_in_idx = np.random.randint(0,4,iters)
I_in_pre = np.array(["intens_flat", "intens_gaussian", "spectrum", "both"])
I_in_conds = I_in_pre[I_in_idx] # which light regime to use

sigmas[I_in_idx == 0] = 0 # no variation of intensity over spectrum

locs[I_in_idx == 1,1] = locs[I_in_idx == 1,0] # same spectrum
sigmas[I_in_idx == 1,1] = sigmas[I_in_idx == 1,0]

luxs[I_in_idx == 2,1] = luxs[I_in_idx == 2,0] # same intensity

# for saving the information
I_in_datas = np.append(locs, np.append(luxs, sigmas,1),1)

# different communities, twice constant light, maximum of those, fluctuating
cases = ["Const1", "Const2", "max_Const", "Fluctuating"]

columns = ['r_pig', 'r_spec', 'r_pig_spec','fac', 'I_in_cond', 'case', 
           'period','pigments'] + list(range(1,11)) \
            + ["loc1", "loc2", "lux1", "lux2", "sigma1", "sigma2"]

data = pd.DataFrame(None,columns = columns, index = range(4*iters))
# getting data from jobscript                    
try:                     
    save = sys.argv[1]
    fluc_case = sys.argv[2]
except IndexError:
    save = np.random.randint(100000)
    fluc_case = "step"
save_string = "data/data_random_"+fluc_case+str(save)+".csv"

# test how long 10 runs go to end programm early enough
test_time_start = timer() 
i = 0
for j in range(10):
    I_ins = np.array([I_in_def(luxs[i,0],locs[i,0],sigmas[i,0]),
            I_in_def(luxs[i,1],locs[i,1],sigmas[i,1])])
    I_in_fun = I_fun.fluc_nconst(I_ins, period = periods[i], fluc_case = fluc_case)
    
    # compute the richnesses
    richnesses = fluctuating_richness(r_pigs[i], r_specs[i], r_pig_specs[i],
            n_com , facs[i], periods[i],pigments[i],I_in_fun, [0,0.5])
    
    # save to dataframe
    for j in range(len(cases)):
        data.iloc[4*i+j] = [r_pigs[i], r_specs[i], r_pig_specs[i], facs[i],
            I_in_conds[i], cases[j],periods[i],pigments[i]]\
            + list(richnesses[j]) + list(I_in_datas[i])
    i+=1
test_time_end = timer()

while timer()-start <3600-(test_time_end-test_time_start):
    # create the light regime
    try:
        I_ins = np.array([I_in_def(luxs[i,0],locs[i,0],sigmas[i,0]),
            I_in_def(luxs[i,1],locs[i,1],sigmas[i,1])])
    except IndexError:
        break
    I_in_fun = I_fun.fluc_nconst(I_ins, period = periods[i], 
                                 fluc_case = fluc_case)
    
    # compute the richnesses
    richnesses = fluctuating_richness(r_pigs[i], r_specs[i], r_pig_specs[i],
            n_com , facs[i], periods[i],pigments[i],I_in_fun, [0,0.5])
    
    # save to dataframe
    for j in range(len(cases)):
        data.iloc[len(cases)*i+j] = [r_pigs[i], r_specs[i], r_pig_specs[i], 
            facs[i],I_in_conds[i], cases[j],periods[i],pigments[i]]\
            + list(richnesses[j]) + list(I_in_datas[i])
    i+=1
data = data[0:i*len(cases)]          
data.to_csv(save_string)
