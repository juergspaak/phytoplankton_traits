"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Create randomized species according to methods
"""

import pandas as pd
import numpy as np
from scipy.integrate import simps

import pigments as lp

# which pyhlum contains which pigment
pig_spe_id = pd.read_csv("Pigment_algae_table.csv")

pigment_id = [] # order of pigments in names_pigments
pigment_id_species = [] # order in Pigment_algae_table.csv
pigment_names = [] # names of all pigments used in the paper

for i,pigment in enumerate(pig_spe_id["Pigment"]):
    if pigment in lp.names_pigments:
        pigment_id.append(lp.names_pigments.index(pigment))
        pigment_id_species.append(i)
        pigment_names.append(pigment)


pigments = lp.pigments[np.array(pigment_id)]
species_all = pig_spe_id.iloc[np.array(pigment_id_species)]
                              
# first two columns are pigment names and where to find which pigment
species_pigments = species_all.iloc[:,2:].values
species_pigments[np.isnan(species_pigments)] = 0
                 
# minus 2, because of pigment labeling and whether pigments are found
n_diff_spe = len(pigments)
                         
def gen_com(present_species, fac, n_com_org = 100, I_ins = None):
    """Generate random species
    
    Generate species with random absorption spectrum according to the table
    of vd Hoek. Species are ensured to survive in monoculture with incoming
    light equal to I_ins
    
    Input:
        present_species: list
            A list containing the identities of each species
        fac: float
            Maximal factor of deviation from parameters in Stomp et al.
        n_com: integer
            number of communities to generate. Actual number of communities
            might differ slightly from this number
        I_ins: `None` or list of incoming lights
            Lights at which each species must be able to survive
            
    Return:
        para: [phi,l]
            The photosynthetic efficiency and loss rate of the species
        k_spec: array with shape (len(lambs), len(present_species),n_com)
            The absorption spectrum of each species in each community
        alphas: array with shape (len(pigments), len(present_species),n_com)
            Concentration of each pigment for each species in each community"""
    # internally generate to many species, as some will not survive
    n_com = n_com_org*10
    
    # species richness to be generated
    r_spec = len(present_species)

    
    # check input
    if max(present_species)>n_diff_spe:
        raise ValueError("maximum of `present_species` must be at most"+
                         "{} entries".format(n_diff_spe))
    
    # photosynthetic efficiency for each species
    phi = 2*1e6*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    # loss rate of the community
    l = 0.014*np.random.uniform(1/fac, 1*fac,(n_com))

    # concentration of each pigment for each species
    alphas = np.random.uniform(1,2,(len(pigments),r_spec,n_com)) *\
                species_pigments[:,present_species, np.newaxis]

    # compute absorption spectrum of each species
    k_spec = np.einsum("pl,psc->lsc",pigments, alphas)
    
    # Total absorption of each species should be equal (similar to Stomp)
    tot_abs = np.mean(simps(k_spec, dx = lp.dlam, axis = 0), axis = -1)
    k_spec = k_spec/tot_abs[:,np.newaxis]*2.0e-7

    # change pigment concentrations accordingly
    alphas = alphas/tot_abs[:,np.newaxis]*2.0e-7
    
    # check survivability in monoculture
    if not(I_ins is None):
        # in some unprobable cases this might generate less than n_com species
        surv = mono_culture_survive(phi/l,k_spec, I_ins)
        n_surv = min(n_com_org, *np.sum(surv, axis = -1))
        spec_id = np.argsort(1-surv,axis = 1)[:,:n_surv]
        dummy_id = np.arange(r_spec).reshape(-1,1)
    else:
        spec_id = np.arange(n_com_org)
        dummy_id = np.arange(r_spec).reshape(-1,1)
        
    # remove species that would not survive
    phi,l = phi[dummy_id, spec_id], l[spec_id]
    k_spec = k_spec[:,dummy_id, spec_id]
    alphas = alphas[:,dummy_id, spec_id]
    
    return np.array([phi,l]), k_spec, alphas

def mono_culture_survive(par, k_spec, I_ins):
    """check whether each species could survive in monoculture
    
    par: phi/l
    k_spec: absorption spectrum
    I_ins: Incoming lights at which species must survive
    
    Returns:
        Surv: boolean array with same shape as par, indicating which species
        survive in all light conditions"""
    # initial growth rate
    I_ins = I_ins.reshape(-1,len(lp.lambs),1,1)
    init_growth = par*simps(I_ins*k_spec,dx = lp.dlam,axis = 1)-1
    # initial growth rate must be larger than 0 for all lights
    survive = np.all(init_growth>0,axis = 0)
    return survive
    

if __name__ == "__main__":
    # For illustration plot the absorption spectrum of some random species
    import matplotlib.pyplot as plt
    
    # Absorption spectrum of all pigments
    fig = plt.figure(figsize=(9,9))
    plt.plot(lp.lambs,pigments.T, label = "1")
    plt.xlabel("nm")
    plt.legend(labels = pigment_names)
    
    # plot the absorption spectrum of random species
    plt.figure()
    par, k_spec, alphas = gen_com(np.random.randint(11,size = 5),4,100)
    plt.plot(k_spec[...,0])   