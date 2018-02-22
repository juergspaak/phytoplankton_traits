"""@author: J.W.Spaak

Computes the number of coexisting species for random settings
uses continuously changing lightspectrum
Incoming light is one gaussian which shifts its peak"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer


from I_in_functions import fluc_continuous
import sys
sys.path.append("../3_different_pigments")
from richness_computation import fluctuating_richness

# getting data from jobscript 
try:                    
    save = sys.argv[1]
    max_r_spec_rich = int(sys.argv[2])
    randomized_pigments = float(sys.argv[3])
    if randomized_pigments:
        save = save +["_randomized_pigments_{}_".format(randomized_pigments)]
except IndexError:
    save = np.random.randint(100000)
    max_r_spec_rich = 11
    randomized_pigments = 0

save_string = "data/data_random_continuous"+str(save)+".csv"

start = timer()
iters = 10000 # number of random settings
n_com = 100 # number of communities in each setting

# make sure, that in each community there are at more pigments than in species
r_pigs_pre = np.random.randint(2,21,2*iters) 
r_pig_specs_pre = np.random.randint(1,max_r_spec_rich,2*iters)
# richness of pigments in community and in each species
r_pigs = r_pigs_pre[r_pigs_pre>r_pig_specs_pre][:iters]
r_pig_specs = r_pig_specs_pre[r_pigs_pre>r_pig_specs_pre][:iters]

r_specs = np.random.randint(1,40,iters) # richness of species
facs= np.random.uniform(1,5,iters) # maximal fitness differences
periods = 10**np.random.uniform(0,2, iters) # ranges from 1 to 1000
pigments = np.random.randint(0,2,iters) 
pigments = np.array(["rand", "real"])[pigments] # real/random pigments

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
cases = ["Const1", "Const2", "Const3", "Const4", "max_Const", "Fluctuating"]

percentile_cols = ["I_out,{}".format(i) for i in ["05",25,50, 75, 95]]+\
                   ["biovolume,{}".format(i) for i in ["05",25,50, 75, 95]]
columns = ['r_pig', 'r_spec', 'r_pig_spec','fac', 'I_in_cond', 'case', 
           'period','pigments'] + list(range(1,11))+percentile_cols \
            + ["loc1", "loc2", "lux1", "lux2", "sigma"]

data = pd.DataFrame(None,columns = columns, index = range(6*iters))

i = 0
# test how long 10 runs go to end programm early enough
test_time_start = timer() 
for j in range(10):
     # create the light regime
    I_in = fluc_continuous(locs[i], luxs[i], periods[i], sigma = sigmas[i])
    # compute the richnesses
    richnesses, I_out, biovolume = fluctuating_richness(r_pigs[i], 
            r_specs[i], r_pig_specs[i],n_com , facs[i], periods[i],pigments[i],
            I_in,np.linspace(0,0.5,4), randomized_pigments)
    # save to dataframe
    for k in range(len(cases)):
        data.iloc[len(cases)*i+k] = [r_pigs[i], r_specs[i], r_pig_specs[i],
            facs[i], I_in_conds[i], cases[k],periods[i],pigments[i]]\
            + list(richnesses[k]) +list(I_out[k])+list(biovolume[k])\
            + list(I_in_datas[:,i])
    i+=1
test_time_end = timer()

while timer()-start <3600-(test_time_end-test_time_start):
    # create the light regime
    try:
        I_in = fluc_continuous(locs[i], luxs[i], periods[i], sigma = sigmas[i])
    except IndexError:
        break
    # compute the richnesses
    richnesses, intens_const, intens_fluct = fluctuating_richness(r_pigs[i], 
            r_specs[i], r_pig_specs[i],n_com , facs[i], periods[i],pigments[i],
            I_in,np.linspace(0,0.5,4), randomized_pigments)
    # save to dataframe
    for k in range(len(cases)):
        data.iloc[len(cases)*i+k] = [r_pigs[i], r_specs[i], r_pig_specs[i],
            facs[i], I_in_conds[i], cases[k],periods[i],pigments[i]]\
            + list(richnesses[k]) +list(intens_const)+list(intens_fluct)\
            + list(I_in_datas[:,i])
    i+=1
data = data[0:i*len(cases)] 
data.to_csv(save_string)
