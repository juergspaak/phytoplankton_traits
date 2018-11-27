"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Create randomized species according to methods
"""

import pandas as pd
import numpy as np
from scipy.integrate import simps
import warnings

# load pigment absorption spectra
df = pd.read_csv("pigments.csv")
lambs = df["lambda"].values
dlam = lambs[1]-lambs[0]                     
pigments = df.values[:,1:].T
pigment_names = df.columns[1:]

# which pigmentation type contains which pigment
pig_spe_id = pd.read_csv("Pigment_algae_table.csv")
species_pigments = pig_spe_id.iloc[:,2:].values
species_pigments[np.isnan(species_pigments)] = 0
                 
n_diff_spe = species_pigments.shape[-1] # number of different species
 
def gen_com(present_species, fac, n_com_org = 100, I_ins = None, 
            k_BG = 0, zm = 100, run = 0):
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
            Concentration of each pigment for each species in each community
            
    units of the returned variables are:
        [phi]: fl * (mu mol photons)^-1
        [l]: h^-1
        [k_spec]: cm^2 * fl^-1
        [alphas]: unitless"""
    if run == 2:
        return None, None, None, None, False
    # internally generate to many species, as some will not survive
    n_com = n_com_org*10
    
    # species richness to be generated
    r_spec = len(present_species)

    
    # check input
    if max(present_species)>n_diff_spe:
        raise ValueError("maximum of `present_species` must be at most"+
                         "{} entries".format(n_diff_spe))
    
    # photosynthetic efficiency for each species
    # unit: [phi] = fl * (mumol photons)^-1
    phi = np.random.uniform(1,3, (r_spec,n_com))*1e6
    # loss rate of the community
    # unit: [l] = h^-1
    l = 0.003*(0.015/0.003)**np.random.uniform(0,1,n_com)

    # concentration of each pigment for each species
    # unit: [alphas] = unitless
    alphas = np.random.uniform(1,2,(len(pigments),r_spec,n_com)) *\
                species_pigments[:,present_species, np.newaxis]

    # compute absorption spectrum of each species
    # unit: [k_spec] = cm^2 * fl^-1
    k_spec = np.einsum("pl,psc->lsc",pigments, alphas)
    
    # Total absorption of each species should be equal (similar to Stomp)
    int_abs = simps(k_spec, dx = dlam, axis = 0)
    k_spec = k_spec/int_abs*2.0e-7

    # change pigment concentrations accordingly
    alphas = alphas/int_abs*2.0e-7
    
    # check survivability in monoculture
    if not(I_ins is None):
        surv = mono_culture_survive(phi/l,k_spec, I_ins,k_BG,zm)
        n_surv = min(n_com_org, sum(surv))
        
        # in some unprobable cases this might generate less than n_com species
        if n_surv == 0:
            return gen_com(present_species, fac, n_com_org, I_ins, k_BG, zm,
                           run+1)
        
        # choose from each species n_surv that survived in all light conditions
        spec_id = np.arange(n_com)[surv][:n_surv]
    else:
        spec_id = np.arange(n_com_org)
    
        
        
    # remove species that would not survive
    phi,l = phi[..., spec_id], l[spec_id]
    k_spec = k_spec[..., spec_id]
    alphas = alphas[..., spec_id]
    
    return phi,l, k_spec, alphas, True

def mono_culture_survive(par, k_spec, I_ins, k_BG = 0 ,zm = 100):
    """check whether each species could survive in monoculture
    
    par: phi/l
    k_spec: absorption spectrum
    I_ins: Incoming lights at which species must survive
    
    Returns:
        Surv: boolean array with same shape as par, indicating which species
        survive in all light conditions"""
    # light condition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # division by 0 is triggered
        light = np.where(zm*k_BG == 0, I_ins, 
                         I_ins*(1-np.exp(-k_BG*zm))/(k_BG*zm))
    light.shape = -1,len(lambs),1,1
    # initial growth rate
    init_growth = par*simps(light*k_spec,dx = dlam,axis = 1)-1
    # initial growth rate must be larger than 0 for all lights
    survive = np.all(init_growth>0,axis = (0,1))
    return survive
    

if __name__ == "__main__":
    # For illustration plot the absorption spectrum of some random species
    import matplotlib.pyplot as plt
    import I_in_functions as I_inf
    
    # Absorption spectrum of all pigments
    fig = plt.figure(figsize=(9,9))
    plt.plot(lambs,pigments.T, label = "1")
    plt.xlabel("nm")
    plt.legend(labels = pigment_names)
    
    # plot the absorption spectrum of random species
    plt.figure()
    phi,l, k_spec, alphas, a  = gen_com(np.random.randint(11,size = 5),4,100,
                        50*I_inf.sun_spectrum["blue sky"], I_inf.k_BG["ocean"])
    plt.plot(k_spec[...,0])   