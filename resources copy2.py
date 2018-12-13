"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Load absorption spectra of pigments found in nature
Load also pigments that are not phytosyntheticall active

Reference papers can be found in the Pigment_algae_table.csv
"""

import numpy as np
from scipy.integrate import simps, odeint
from phytoplankton_communities.generate_species import pigments, lambs, dlam
from phytoplankton_communities.I_in_functions import zm
import matplotlib.pyplot as plt

pig_id = [0,6]

n_pig_conc = 4
n_phi = 4
pig_conc = np.linspace(1-1e-3,0.5,n_pig_conc)

# concentration of each pig in each species
alphas = np.zeros((9,2,n_pig_conc))

alphas[pig_id,0] = np.array([pig_conc,1-pig_conc])
alphas[pig_id,1] = 1- alphas[pig_id,0]

# create the absorption spectrum of the species
k_spec = np.einsum("pl,psc->lsc",pigments, alphas)
    
# Total absorption of each species should be equal (similar to Stomp)
int_abs = simps(k_spec, dx = dlam, axis = 0)
k_spec = k_spec/int_abs*2.0e-7

# change pigment concentrations accordingly
alphas = alphas/int_abs*2.0e-7

l = 0.01
phi = np.full((2,1,n_phi), 2e6)

fac_phi = 2**(np.linspace(-1,1,n_phi)/2)
phi[0] *= fac_phi
phi[1] /= fac_phi

def multi_growth(N,t,I_in):
    # reshape for odeint
    N = N.reshape(-1,n_phi, n_pig_conc)
    # growth rate of the species
    # sum(N_j*k_j(lambda))
    tot_abs = zm*(np.nansum(N*k_spec[...,None], axis = 1, keepdims = True))
    # growth part
    growth = phi*simps(k_spec[...,None]/tot_abs*(1-np.exp(-tot_abs))\
                       *I_in(t).reshape(-1,1,1,1),dx = dlam, axis = 0)
                       
    return (N*(growth-l)).reshape(-1)


n_I_in = 1000
time = (24*100)**np.linspace(0,1,10)
I_ins = np.empty((n_I_in,len(lambs)))
N_all = np.empty((n_I_in,len(time),2,n_pig_conc, n_phi))

for i in range(n_I_in):
    print(i)
    I_in = np.exp(-(np.random.uniform(450,550)-lambs)**2
                  /np.random.uniform(50,500)**2)
    I_in = I_in/simps(I_in, dx = dlam)*np.random.uniform(20,100)
    I_in_t = lambda t: I_in
    I_ins[i] = I_in
    N0 = np.full(2*n_pig_conc* n_phi, 1e8)


    N_t = odeint(multi_growth, N0, time, args = (I_in_t,))

    N_all[i] = N_t.reshape((-1,2,n_pig_conc, n_phi))


# compute the contribution of both pigments
pigs = pigments[pig_id]


resources = simps(pigs[:,np.newaxis]*I_ins,dx = dlam, axis = -1)


pig1 = np.linspace(0,2000)
pig2 = (l/phi-pig1.reshape(-1,1,1,1)*alphas[pig_id[0],...,None])\
                /alphas[pig_id[1],...,None]

fig, ax = plt.subplots(n_pig_conc, n_phi, figsize = (12,12), 
                       sharex = True, sharey = True)
cm = plt.cm.get_cmap('RdYlBu')
for i in range(n_pig_conc):
    for j in range(n_phi):
        axc = ax[i,j]
        color = N_all[:,-1,0]/np.sum(N_all[:,-1],axis = 1)
        cmap = axc.scatter(resources[0].flatten(), resources[1].flatten(), 
            c = color[:,i,j], linewidth = 0, vmin = 0, vmax = 1, s = 10)
        axc.plot(pig1,pig2[...,i,j], linewidth = 2)
        
plt.axis([0,None, 0,2000])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cmap, cax=cbar_ax)

fig.savefig("Example of resource use.pdf")