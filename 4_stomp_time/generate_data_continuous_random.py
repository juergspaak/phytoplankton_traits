"""@author: J.W.Spaak

Computes the number of coexisting species for random settings
uses continuously changing lightspectrum"""

import pandas as pd
import numpy as np

from fluctuating_spectra import fluctuating_richness
from I_in_functions import fluc_continuous

import sys
sys.path.append("../3_different_pigments")


iters = 100000 # number of random settings
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
sigmas = 2**np.random.uniform(4,9,iters) # ragnes from 16-512
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

columns = ['r_pig', 'r_spec', 'r_pig_spec','fac', 'I_in_cond', 'case', 
           'period','pigments'] + list(range(1,11)) \
            + ["loc1", "loc2", "lux1", "lux2", "sigma"]

data = pd.DataFrame(None,columns = columns, index = range(6*iters))
# save with random number to avoid etasing previous files                        
save = np.random.randint(100000) 
save_string = "data_continuous_random,"+str(save)+".csv"

for i in range(iters):
    # create the light regime
    I_in = fluc_continuous(locs[i], luxs[i], periods[i], sigma = sigmas[i])
    # compute the richnesses
    richnesses = fluctuating_richness(r_pigs[i], r_specs[i], r_pig_specs[i],
            n_com , facs[i], periods[i],pigments[i],I_in,np.linspace(0,0.5,4))
    # save to dataframe
    for j in range(len(cases)):
        data.iloc[6*i+j] = [r_pigs[i], r_specs[i], r_pig_specs[i], facs[i],
            I_in_conds[i], cases[j],periods[i],pigments[i]]\
            + list(richnesses[j]) + list(I_in_datas[:,i])
    if i%1000 == 999: # save to not lose progress
        data.to_csv(save_string)
          

#data.to_csv(save_string)
