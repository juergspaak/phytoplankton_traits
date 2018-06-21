"""create realistic species

uses the table of the EAWAG"""

import pandas as pd
import numpy as np

import load_pigments as lp
from scipy.integrate import simps

# to reproduce exact values of paper

pig_spe_id = pd.read_csv("vdHoek,pig_alg_no_protection.csv")

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
                         
def gen_com(present_species, fac, n_com_org = 100, end = 2e-7,case = 0,
            I_ins = None, no_super = False):
    """Generate species for the spectrum model
    
    Generate species with random absorption spectrum according to the table
    of vd Hoek. Species are ensured to survive in monoculture with incoming
    light equal to I_ins
    
    Input:
        present_species: list
            A list containing the identities of each species
        fac: float
            Maximal facto of deviation from parameters in Stomp et al.
        n_com: integer
            number of communities to generate. Actual number of communities
            might differ slightly from this number
        end: float
            Average of total absorption, default is equal to stomp values
        case: integer in [0,1,2]
            Which pigments of vd Hoek shall be used to generate the absorption
            spectum. 0-> all pigments, 1 -> only abundant pigments,
            2-> only the main pigments of the species
        I_ins: `None` or list of incoming lights
            Lights at which each species must be able to survive
        no_super: bool, default is False
            If True, then all species will have the same total absorption.
            If False (default), then the species on average will have the same
            total absorption
            
    Return:
        para: [phi,l]
            The photosynthetic efficiency and loss rate of the species
        k_spec: array with shape (len(lambs), len(present_species),n_com)
            The absorption spectrum of each species in each community
        alphas: array with shape (len(pigments), len(present_species),n_com)
            Concentration of each pigment for each species in each community"""
    # internally generate to many species
    n_com = n_com_org*10
    
    r_spec = len(present_species)
    
    # only use most important pigments
    if case==1:
        species_pigments[species_pigments<0.5] = 0
    if case ==2:
        species_pigments[species_pigments<1] = 0
    
    # check input
    if max(present_species)>n_diff_spe:
        raise ValueError("maximum of `present_species` must be at most"+
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

    # don't allow super species, i.e. all species have same total absorption
    if no_super:
        alphas = alphas/simps(k_spec, dx=lp.dlam, axis = 0)*end
        k_spec = k_spec/simps(k_spec, dx=lp.dlam, axis = 0)*end

    if not(I_ins is None):
        surv = mono_culture_survive(phi/l,k_spec, I_ins)
        n_surv = min(n_com_org, *np.sum(surv, axis = -1))
        spec_id = np.argsort(1-surv,axis = 1)[:,:n_surv]
        dummy_id = np.arange(r_spec).reshape(-1,1)
    else:
        spec_id = np.arange(n_com_org)
        dummy_id = np.arange(r_spec).reshape(-1,1)
    phi,l = phi[dummy_id, spec_id], l[dummy_id, spec_id]
    k_spec = k_spec[:,dummy_id, spec_id]
    alphas = alphas[:,dummy_id, spec_id]
    if not(I_ins is None):
        surv = mono_culture_survive(phi/l,k_spec, I_ins)
        if np.any(np.logical_not(surv)):
            raise
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
    import matplotlib.pyplot as plt
    np.random.seed(5)
    par, k_spec, alphas = gen_com(np.arange(10),4,100,
                                  I_ins = np.array([I_in_ref(0),I_in_ref(5)]))
    plt.plot(k_spec[...,0])
    
    
    surv = mono_culture_survive(par[0]/par[1], k_spec, 
                                   np.array([I_in_ref(0),I_in_ref(5)]))
    

    surv = mono_culture_survive(par[0]/par[1],k_spec, np.array([I_in_ref(0),I_in_ref(5)]))
    n_surv = min(n_com_org, *np.sum(surv, axis = -1))
    spec_id = np.argsort(1-surv,axis = 1)[:,:n_surv]
    