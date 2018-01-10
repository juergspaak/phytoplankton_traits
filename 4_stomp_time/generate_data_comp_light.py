"""@author: J.W.Spaak

Computes the number of coexisting species for random settings"""

import pandas as pd
import numpy as np

from fluctuating_spectra import fluctuating_richness
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

n_peaks = np.random.randint(5,20,iters)

# different communities, twice constant light, maximum of those, fluctuating
cases = ["Const1", "Const2", "max_Const", "Fluctuating"]

columns = ['r_pig', 'r_spec', 'r_pig_spec','fac', 'n_peaks','kl_div1','kl_div2'
           ,'case', 'period','pigments'] + list(range(1,11))

data = pd.DataFrame(None,columns = columns, index = range(4*iters))
# getting data from jobscript                    
try:                     
    save = sys.argv[1]
except IndexError:
    save = np.random.randint(100000)
save_string = "data_random_comp_light"+str(save)+".csv"

# test how long 10 runs go to end programm early enough
test_time_start = timer() 
i = 0
for j in range(10):
    I_in1,kl_div1 = I_fun.I_in_composite(n_peaks[i])
    I_in2,kl_div2 = I_fun.I_in_composite(n_peaks[i])
    I_in_fun = I_fun.fluc_nconst([I_in1,I_in2], period = periods[i])
    
    # compute the richnesses
    richnesses = fluctuating_richness(r_pigs[i], r_specs[i], r_pig_specs[i],
            n_com , facs[i], periods[i],pigments[i],I_in_fun, [0,0.5])
    
    # save to dataframe
    for j in range(len(cases)):
        data.iloc[4*i+j] = [r_pigs[i], r_specs[i], r_pig_specs[i], facs[i],
            n_peaks[i],kl_div1,kl_div2, cases[j],periods[i],pigments[i]]\
            + list(richnesses[j])
    i+=1
test_time_end = timer()

while timer()-start <3600-(test_time_end-test_time_start):
    # create the light regime
    try:
        I_in1,kl_div1 = I_fun.I_in_composite(n_peaks[i])
    except IndexError:
        break
        
    I_in2,kl_div2 = I_fun.I_in_composite(n_peaks[i])
    I_in_fun = I_fun.fluc_nconst([I_in1,I_in2], period = periods[i])
    
    # compute the richnesses
    richnesses = fluctuating_richness(r_pigs[i], r_specs[i], r_pig_specs[i],
            n_com , facs[i], periods[i],pigments[i],I_in_fun, [0,0.5])
    
    # save to dataframe
    for j in range(len(cases)):
        data.iloc[4*i+j] = [r_pigs[i], r_specs[i], r_pig_specs[i], facs[i],
            n_peaks[i],kl_div1,kl_div2, cases[j],periods[i],pigments[i]]\
            + list(richnesses[j])
    i+=1
data = data[0:i*len(cases)]          
data.to_csv(save_string)
