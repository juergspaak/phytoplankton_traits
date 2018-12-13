"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Similar to sim_richness.py
Differences are:
    Nonzero background absorptivity
    Pigments are slightly different in different spieces

generates the data data_ap_richness*.csv that is used in plot_ap_richness.py"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer

from phytoplankton_communities.I_in_functions import sun_light, k_BG
from phytoplankton_communities.generate_species import n_diff_spe,pigments
import sys
import phytoplankton_communities.richness_computation as rc

# getting data from jobscript 
try:                    
    save = sys.argv[1]
    np.random.seed(int(save))
except IndexError:
    save = np.random.randint(100000)

save_string = "data/data_ap_richness"+str(save)+".csv"

start = timer()
iters = 10000 # number of random settings
n_com = 100 # number of communities in each setting


r_specs = np.random.randint(1,16,iters) # richness of species
facs = np.random.uniform(1.5,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 100

# Determining the light regime for each setting
# random incoming light fluctuations
luxs = np.random.uniform(20,200,(iters,2))


# different communities, twice constant light, maximum of those, fluctuating
cases = ["Const1", "Const2", "Const3", "Const4", "Fluctuating"]

prob_cols = ["spec_rich,{}".format(i) for i in range(6)] \
            + ["pig_rich,{}".format(i) for i in range(len(pigments)+1)]

EF_cols = ["biovolume,{}".format(i) for i in ["05",25,50, 75, 95]]
columns = ["case","species","fac","period","lux1", "lux2", "r_pig_start",
           "r_pig_equi", "r_spec_equi"]+EF_cols+ prob_cols +["n_fix"]

data = pd.DataFrame(None,columns = columns, index = range(len(cases)*iters))
    
i = 0
# test how long 10 runs go to end programm early enough
test_time_start = timer() 

time_for_10 = 0

while timer()-start <3600-(time_for_10):
    if i==iters:
        break
    I_in = sun_light(luxs[i], periods[i])
    present_species = np.random.choice(n_diff_spe, r_specs[i],replace = True)
    # compute the richnesses
    (richness_equi, EF_biovolume, r_pig_equi, r_pig_start, prob_spec, 
            prob_pig, n_fix) = rc.fluctuating_richness(present_species, 
            n_com , facs[i], periods[i],I_in,np.linspace(0,0.5,4),
            randomized_spectra = 0.05, k_BG = k_BG["baltic sea"])
    print(i)
    # save to dataframe
    for k,case in enumerate(cases):
        data.iloc[len(cases)*i+k] = [cases[k],str(present_species), facs[i], 
            periods[i], *luxs[i], r_pig_start, r_pig_equi[k], richness_equi[k], 
            *EF_biovolume[k], *prob_spec[k], *prob_pig[k], n_fix]
    i+=1
    if i==10:
        time_for_10 = timer()-test_time_start
data = data[0:i*len(cases)] 
data.to_csv(save_string)
