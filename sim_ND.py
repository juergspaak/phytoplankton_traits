"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Computes the number of coexisting species for random settings
uses continuously changing lightspectrum
Incoming light is one gaussian which shifts its peak

generates the data data_richness*.csv that is used in plot_richness.py"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer

from phytoplankton_communities.I_in_functions import sun_spectrum
from phytoplankton_communities.generate_species import n_diff_spe,pigments
import sys
import phytoplankton_communities.ND_computation as nc

# getting data from jobscript 
try:                    
    save = sys.argv[1]
    # to reproduce the exacte data used in the paper
    np.random.seed(int(save))
    iters = 10000 # number of random settings
except IndexError:
    save = np.random.randint(10000,100000)
    iters = 100

save_string = "data/data_ND"+str(save)+".csv"

start = timer()

n_com = 100 # number of communities in each setting


r_specs = np.random.randint(1,16,iters) # richness of species
facs = np.random.uniform(1.5,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 100

# Determining the light regime for each setting
# random incoming light fluctuations
luxs = np.random.uniform(20,200,(iters))
equi_cols = ["equi_{}".format(i) for i in range(1,6)]
NFD_cols = ["ND_{}".format(i) for i in range(1,6)] + ["FD_{}".format(i)
    for i in range(1,6)]
columns = ["species","fac","lux", "r_pig_start", "r_pig_equi", "r_spec_equi",
           "selection", "complementarity", "EF"]+equi_cols + NFD_cols

data = pd.DataFrame(None,columns = columns, index = range(n_com*iters),
                    dtype = float)
    
i = 0
# test how long 10 runs go to end programm early enough
test_time_start = timer()

time_for_10 = 0
counter = 0
while timer()-start <3600-(time_for_10):
    if i==iters:
        break
    I_in = luxs[i] * sun_spectrum["blue sky"]
    present_species = np.random.choice(n_diff_spe, r_specs[i],replace = True)
    # compute the richnesses
    computed_data = nc.constant_richness(present_species, n_com, facs[i],
                    I_in = I_in)
    print(i, "iteration")
    if computed_data is None:
        continue
    n_fix = len(computed_data)
    primary_data = np.tile([facs[i], luxs[i]], (n_fix,1))
    data.iloc[counter: (counter + n_fix), 1:] = np.append(primary_data,
              computed_data ,axis = 1)
    data.iloc[counter: (counter + n_fix), 0] = str(present_species)
    counter += n_fix
    i+=1
    if i==10:
        time_for_10 = timer()-test_time_start
data = data[0:counter]
data.to_csv(save_string)
