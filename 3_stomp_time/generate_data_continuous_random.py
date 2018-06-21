"""@author: J.W.Spaak

Computes the number of coexisting species for random settings
uses continuously changing lightspectrum
Incoming light is one gaussian which shifts its peak"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer

from I_in_functions import fluc_continuous
from generate_species import n_diff_spe,pigments
import sys
sys.path.append("../3_different_pigments")
import richness_computation as rc

# getting data from jobscript 
try:                    
    save = sys.argv[1]
    randomized_pigments = float(sys.argv[2])
    if randomized_pigments:
        save = "_randomized_pigments_{}".format(randomized_pigments) + save
    no_super = sys.argv[3] == "True"
    if no_super:
        save = "_no_super" + save 
except IndexError:
    save = np.random.randint(100000)
    randomized_pigments = 0

save_string = "data/data_EF_fluct"+str(save)+".csv"

start = timer()
iters = 10000 # number of random settings
n_com = 100 # number of communities in each setting


r_specs = np.random.randint(1,15,iters) # richness of species
facs = np.random.uniform(1,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 100

## Determining the light regime for each setting
# random incoming light fluctuations
luxs = np.random.uniform(30,100,(iters,2))
sigmas = 2**np.random.uniform(5,9,iters) # ragnes from 16-512
locs = np.random.uniform(450,650,(iters,2))

# for saving the information
I_in_datas = np.empty((5,iters))
I_in_datas[0:2] = locs.T
I_in_datas[2:4] = luxs.T
I_in_datas[4] = sigmas

# different communities, twice constant light, maximum of those, fluctuating
cases = ["Const1", "Const2", "Const3", "Const4", "Fluctuating"]

prob_cols = ["spec_rich,{}".format(i) for i in range(6)] \
            + ["pig_rich,{}".format(i) for i in range(len(pigments)+1)]

EF_cols = ["biovolume,{}".format(i) for i in ["05",25,50, 75, 95]]
columns = ["case","species","fac","period","loc1",
            "loc2","lux1", "lux2", "sigma", "r_pig_start", "r_pig_equi", 
           "r_spec_equi"]+EF_cols+ prob_cols

data = pd.DataFrame(None,columns = columns, index = range(len(cases)*iters))
    
i = 0
# test how long 10 runs go to end programm early enough
test_time_start = timer() 

time_for_10 = 0

while timer()-start <3600-(time_for_10):
    if i==iters:
        break
    I_in = fluc_continuous(locs[i], luxs[i], periods[i], sigma = sigmas[i])
    present_species = np.random.choice(n_diff_spe, r_specs[i],replace = True)
    # compute the richnesses
    (richness_equi, EF_biovolume, r_pig_equi, r_pig_start, prob_spec, 
            prob_pig) = rc.fluctuating_richness(present_species, 
            n_com , facs[i], randomized_pigments, periods[i],
            I_in,np.linspace(0,0.5,4), no_super)
    print(i)
    # save to dataframe
    for k,case in enumerate(cases):
        data.iloc[len(cases)*i+k] = [cases[k],str(present_species), facs[i], 
                  periods[i], *locs[i], *luxs[i], sigmas[i],
                r_pig_start, r_pig_equi[k], richness_equi[k], 
                *EF_biovolume[k], *prob_spec[k], *prob_pig[k]]
    i+=1
    if i==10:
        time_for_10 = timer()-test_time_start
data = data[0:i*len(cases)] 
data.to_csv(save_string)
