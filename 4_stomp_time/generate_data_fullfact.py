"""@author: J.W.Spaak

Computes the number of coexisting species for the full factorial setting"""

import pandas as pd
import numpy as np

from fluctuating_spectra import fluctuating_richness

import sys
sys.path.append("../3_different_pigments")
from multispecies_functions import I_in_def

n_com = 100 # number of communities for each factorial combination
pig = "real" # use random or real pigments
period = 10 # period length of fluctuation
r_pigs = np.arange(2,22,2) # number of pigments in the community
r_specs = np.array([10,5,20,40]) # number of species in regional species pool
r_pig_specs = np.arange(1,11,2) # richness of pigments in each species
facs = np.array([2,1.5,5,1.1]) # maxiaml factor of scaled fitness diff

# different light conditions
# intens: fluctuate light intensity, uniform spectrum
# spectrum: fluctuate only spectrum
# both: fluctuate spectrum and intensity
# intens2: fluctuate only intensity, gaussian spectrum
I_in_conds = ["intens", "spectrum", "both", "intens2"]
I_ins = {"intens":[I_in_def(40/300), I_in_def(60/300)],
         "intens2":[I_in_def(40/300, 450, 50), I_in_def(60/300,450,50)],
         "spectrum": [I_in_def(40/300, 450,50), I_in_def(40/300, 650, 50)],
         "both": [I_in_def(40/300, 450, 50), I_in_def(60/300, 650,50)]}

# different communities, twice constant light, maximum of those, fluctuating
cases = ["Const1", "Const2", "max_Const", "Fluctuating"]

# colums of csv file for saving
columns = ['r_pig', 'r_spec', 'r_pig_spec','fac', 'I_in', 'case']\
                            +list(range(1,11))
len_index = len(cases)*len(r_pigs)*len(r_specs)*len(r_pig_specs)*\
            len(facs)*len(I_ins)
data = pd.DataFrame(None,columns = columns, index = range(len_index))
counter = 0
# save with random number to avoid etasing previous files                        
save = np.random.randint(100000) 
save_string = "data/data_period_"+str(period)+","+str(pig)+","+str(save)+".csv"
# full factorial setup
for fac in facs:
    for r_spec in r_specs:
        for r_pig_spec in r_pig_specs:
            data.to_csv("Prelim, "+save_string) # to not lose progress
            for r_pig in r_pigs:
                for I_in_cond in I_in_conds:
                    richnesses = np.nan*np.ones((4,10))
                    if r_pig>=r_pig_spec:
                        try:
                            richnesses = fluctuating_richness(r_pig, r_spec, 
                                r_pig_spec , n_com , fac, period,pig,
                                        I_ins[I_in_cond])
                        except:
                            pass
                    for i in range(len(cases)):
                        data.iloc[counter] = [r_pig, r_spec, r_pig_spec, fac,
                            I_in_cond, cases[i]]+list(richnesses[i])
                        counter += 1

data.to_csv(save_string)