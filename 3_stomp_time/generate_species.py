"""create realistic species

uses the table of the EAWAG"""

import pandas as pd
import numpy as np

import load_pigments as lp
from scipy.integrate import simps

# to reproduce exact values of paper

pig_spe_id = pd.read_csv("vdHoek,pig_alg.csv")

pigment_id = []
pigment_id_species = []
pigment_names = []

for i,pigment in enumerate(pig_spe_id["Pigment"]):
    if pigment in lp.names_pigments:
        pigment_id.append(lp.names_pigments.index(pigment))
        pigment_id_species.append(i)
        pigment_names.append(pigment)
        
pigments = lp.pigments[np.array(pigment_id)]
species_all = pig_spe_id.iloc[np.array(pigment_id_species)]
species_pigments = species_all.iloc[:,2:].values
species_pigments[np.isnan(species_pigments)] = 0
# minus 2, because of pigment labeling and whether pigments are found
n_diff_spe = species_all.shape[-1]-2
                         
def gen_com(present_species, fac, n_com = 100, end = 2e-7,case = 0):
    # additional assumption 1: all pigments have same concnetrations
    r_spec = len(present_species)
    if case ==2:
        species_pigments[species_pigments<1] = 0
    if case==1:
        species_pigments[species_pigments<0.5] = 0
    # check input
    if max(present_species)>n_diff_spe:
        raise ValueError("length of `present_species` must be at most"+
                         "{} entries".format(n_diff_spe))
    # growth parameters of each species
    phi = 2*1e6*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))

    # concentration of each pigment for each species
    alphas = np.random.uniform(1,2,(len(pigments),r_spec,n_com)) *\
                species_pigments[:,present_species, np.newaxis]
    
    # compute absorption spectrum of each species
    k_spec = np.einsum("pl,psc->lsc",pigments, alphas)
    
    # average of species should have similar total absorption as in stomp
    av_tot_abs = np.mean(simps(k_spec, dx = lp.dlam, axis = 0), axis = -1)
    k_spec = k_spec/av_tot_abs[:,np.newaxis]*end

    # change pigment concentrations accordingly
    alphas = alphas/av_tot_abs[:,np.newaxis]*end
    return np.array([phi,l]), k_spec, alphas

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(5)
    par, k_spec, alphas = gen_com(np.arange(10),4,100)
    plt.plot(k_spec[...,0])