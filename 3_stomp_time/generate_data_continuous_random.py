"""@author: J.W.Spaak

Computes the number of coexisting species for random settings
uses continuously changing lightspectrum
Incoming light is one gaussian which shifts its peak"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer

from I_in_functions import fluc_continuous
from generate_species import n_diff_spe
import sys
sys.path.append("../3_different_pigments")
import richness_computation as rc

# getting data from jobscript 
try:                    
    save = sys.argv[1]
    randomized_pigments = float(sys.argv[2])
    if randomized_pigments:
        save = save +["_randomized_pigments_{}_".format(randomized_pigments)]
except IndexError:
    save = np.random.randint(100000)
    max_r_spec_rich = 11
    randomized_pigments = 0

save_string = "data/data_EF"+str(save)+".csv"

start = timer()
iters = 10000 # number of random settings
n_com = 100 # number of communities in each setting


r_specs = np.random.randint(1,40,iters) # richness of species
facs= np.random.uniform(1,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 1000

## Determining the light regime for each setting
# random incoming light fluctuations
luxs = np.random.uniform(30/300,100/300,(iters,2))
sigmas = 2**np.random.uniform(5,9,iters) # ragnes from 16-512
locs = np.random.uniform(450,650,(iters,2))

I_in_idx = np.random.randint(0,2,iters)
I_in_pre = np.array(["spectrum", "both"])
I_in_conds = I_in_pre[I_in_idx] # which light regime to use

luxs[I_in_idx == 0,1] = luxs[I_in_idx == 0,0] # same intensity

# for saving the information
I_in_datas = np.empty((5,iters))
I_in_datas[0:2] = locs.T
I_in_datas[2:4] = luxs.T
I_in_datas[4] = sigmas

# different communities, twice constant light, maximum of those, fluctuating
cases = ["Const1", "Const2", "Const3", "Const4", "Fluctuating"]


EF_cols = ["Pigment concentration",
                   *["biovolume,{}".format(i) for i in ["05",25,50, 75, 95]]]
species_cols = ["survive {}".format(i) for i in range(n_diff_spe)]
columns = ["case","species","r_spec", "fac","period","I_in_cond","loc1",
            "loc2","lux1", "lux2", "sigma", "r_pig_start", "r_pig_equi", 
           "r_spec_equi"]+species_cols+EF_cols

data = pd.DataFrame(None,columns = columns, index = range(len(cases)*iters))

def fill_data(i):
    print(i)
    I_in = fluc_continuous(locs[i], luxs[i], periods[i], sigma = sigmas[i])
    predef_spe = min(n_diff_spe, np.random.randint(r_specs[i])+1)
    present_species = np.random.choice(n_diff_spe, predef_spe)
    # compute the richnesses
    (richness_equi, EF_biovolume, r_pig_equi, EF_pigment, r_pig_start,
            surviving_species) = rc.fluctuating_richness(present_species, 
            r_specs[i], n_com , facs[i], randomized_pigments, periods[i],
            I_in,np.linspace(0,0.5,4))
    # save to dataframe
    for k,case in enumerate(cases):
        data.iloc[len(cases)*i+k] = [case,present_species, r_specs[i], facs[i], 
                  periods[i], I_in_conds[i], *locs[i], *luxs[i], sigmas[i],
                r_pig_start, r_pig_equi[k], richness_equi[k], 
                *surviving_species[k],EF_pigment[k],*EF_biovolume[k]]
i = 0
# test how long 10 runs go to end programm early enough
test_time_start = timer() 
for j in range(10):
    fill_data(i)
    i+=1
test_time_end = timer()

while timer()-start <3600-(test_time_end-test_time_start):
    if i==iters:
        break
    fill_data(i)
    i+=1
data = data[0:i*len(cases)] 
data.to_csv(save_string)
