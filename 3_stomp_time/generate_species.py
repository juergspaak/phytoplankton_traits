"""create realistic species

uses the table of the EAWAG"""

import pandas as pd
import numpy as np

import load_pigments as lp

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
n_diff_spe = species_all.shape[-1]-1
                          
def gen_com(present_species, r_spec, fac, n_com = 100):
    # check input
    if r_spec < len(present_species):
        raise ValueError("`r_spec` must be larger than `present_species`")
    if len(present_species)>n_diff_spe:
        raise ValueError("length of `present_species` must be at most"+
                         "{} entries".format(n_diff_spe))
    # growth parameters of each species
    phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
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
    k_spec /= np.sum(alphas,axis = 0) # normalize expected pigment concentration
    return np.array([phi,l]), k_spec, alphas, species_id

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    par, k_spec, alphas, species_id = gen_com(np.arange(5),10,4,100)
    plt.plot(k_spec[...,0])