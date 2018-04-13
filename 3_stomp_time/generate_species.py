"""create realistic species

uses the table of the EAWAG"""

import pandas as pd
import numpy as np

import load_pigments as lp
from scipy.integrate import simps

np.random.seed(22061991)

pig_spe_id = pd.read_csv("pig_spe_id.csv",delimiter = ",")


pigment_id = []
pigment_id_species = []

for i,pigment in enumerate(pig_spe_id["Pigment"]):
    if pigment in lp.names_pigments:
        pigment_id.append(lp.names_pigments.index(pigment))
        pigment_id_species.append(i)
        
pigments = lp.pigments[np.array(pigment_id)]
species_all = pig_spe_id.iloc[np.array(pigment_id_species)]
species_pigments = species_all.iloc[:,1:].values
# minus 1, because of pigment labeling
n_diff_spe = species_all.shape[-1]-1


                          
def gen_com(present_species, fac, n_com = 100, assumption = [0,0,0]):
    # additional assumption 1: all pigments have same concnetrations
    r_spec = len(present_species)
    if assumption[0]:
        species_pigments[species_pigments>0] = 1
    # check input
    if len(present_species)>n_diff_spe:
        raise ValueError("length of `present_species` must be at most"+
                         "{} entries".format(n_diff_spe))
    # growth parameters of each species
    phi = 2*1e6*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    
    # assign a species type to each species
    species_id = np.random.choice(present_species, (r_spec, n_com))
    # ensure, that each present species occurs at least once
    species_id[:len(present_species)] = present_species[:,np.newaxis]

    # concentration of each pigment for each species
    alphas = np.random.uniform(1,2,(len(pigments),r_spec,n_com)) *\
                species_pigments[:,species_id]
    
    # compute absorption spectrum of each species
    k_spec = np.einsum("pl,psc->lsc",pigments, alphas)
    # Assumption 2: Don't normalize with pigment number
    if assumption[1]:
        k_spec /= np.sum(alphas,axis = 0) # normalize expected pigment concentration
    if assumption[2]:
        # Assumption 3: Total absorption of all species is equal
        k_spec = k_spec/simps(k_spec, dx = lp.dlam, axis = 0)
    # average of species should have similar total absorption as in stomp
    k_spec = k_spec/np.average(simps(k_spec, dx = lp.dlam, axis = 0))*1.5e-7
    return np.array([phi,l]), k_spec, alphas, species_id

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    par, k_spec, alphas, species_id = gen_com(np.arange(5),10,4,100)
    plt.plot(k_spec[...,0])
    plt.figure()
    import pandas as pd
    itera = int(1e4)
    df = pd.DataFrame(None, columns = ["species", "tot_abs"], 
                      index = range(n_diff_spe*itera))
    df.iloc[:,0] = np.repeat(np.arange(n_diff_spe,dtype = float), itera)
    for i in range(n_diff_spe-1):
        par, k_spec, alphas, specis_id = gen_com(np.array([i,i+1]), 10,4,1000, [0,0,0])
        df.iloc[i*itera:(i+1)*itera,1] = simps(k_spec, dx = 3, axis = 0).reshape(-1)
        
    sns.violinplot(x = "species", y = "tot_abs", data = df.convert_objects(convert_numeric = True))