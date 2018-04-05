"""create realistic species

uses the table of the EAWAG"""

import pandas as pd
import numpy as np

from load_pigments import lambs

pig_spe_id = pd.read_csv("pig_spe_id.csv",delimiter = ",")

# load pigments from küpper
file = "../../2_data/3. Different pigments/gp_krueger.csv"

gp_data = pd.read_csv(file)
absorptivity = pd.read_csv(file[:35]+"absorptivity_Krueger.csv")

a = gp_data.iloc[::3,2:].values
xp = gp_data.iloc[1::3, 2:].values
sig = gp_data.iloc[2::3,2: ].values

kuepper = np.nansum(a*np.exp(-0.5*((xp-lambs.reshape(-1,1,1))/sig)**2),-1).T
kuepper *= 1e-8*absorptivity.iloc[:,1].reshape(-1,1)

pigment_id_kuep = []
pigment_id_species = []

kuep_pigments = absorptivity["Pigment"].values
ind_kuep = np.arange(len(kuep_pigments))

for i,pigment in enumerate(pig_spe_id["Pigment"]):
    if pigment in list(absorptivity["Pigment"]):
        pigment_id_kuep.append(ind_kuep[pigment == kuep_pigments])
        pigment_id_species.append(i)
        
pigments = kuepper[np.array(pigment_id_kuep)][:,0]
species_all = pig_spe_id.iloc[np.array(pigment_id_species)]
species_pigments = species_all.iloc[:,1:].values
n_diff_spe = species_all.shape[-1]-1
                          
def gen_com(r_spec, fac, n_com = 100):
    # growth parameters of each species
    phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    
    # assign a species type to each species
    species_id = np.random.choice(n_diff_spe, (r_spec, n_com))
    new = species_pigments[:,species_id]
    alphas = np.random.uniform(1,2,(len(pigments),r_spec,n_com))
    
    k_spec = np.einsum("pl,psc->lsc",pigments, alphas*new)
    k_spec /= np.sum(new,axis = 0)
    return np.array([phi,l]), k_spec, alphas*new