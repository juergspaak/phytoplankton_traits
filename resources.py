"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Load absorption spectra of pigments found in nature
Load also pigments that are not phytosyntheticall active

Reference papers can be found in the Pigment_algae_table.csv
"""

import numpy as np
from scipy.integrate import simps, odeint
from phytoplankton_communities.generate_species import gen_com, pigments
from phytoplankton_communities.I_in_functions import sun_spectrum, k_BG, zm, dlam
from phytoplankton_communities.richness_computation import multi_growth
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#np.random.seed(hash("hello")%4294967295)

I_in = sun_spectrum["usual"]*40
I_in_t = lambda t: I_in
phi,l, k_spec, alphas,a = gen_com([4,4], 1, 20)
l[:] = np.mean(l)

k_key = "clear"
N0 = np.full(phi.size, 1e7)
time = (24*1000)**np.linspace(0,1)

N_t = odeint(multi_growth, N0, time, args = (I_in_t,k_spec, phi,l,zm,
                        k_BG[k_key].reshape(-1,1,1), True))

N_t.shape = (len(time),) + phi.shape

plt.figure()
plt.title("densities over time")
plt.xlabel("time [days]")
plt.ylabel("densities of the species")
plt.plot(time/24, N_t[...,0])

# compute the contribution of both pigments
pigs = pigments[:2]
pigs.shape = len(pigs),1,-1,1

def pig_contribution(pigs):
    tot_abs = zm*(np.nansum(N_t[:,np.newaxis]*k_spec, axis = 2)
                +k_BG[k_key].reshape(-1,1))
    return simps(pigs/tot_abs*I_in[:,np.newaxis]*(1-np.exp(-tot_abs)),
                 dx = dlam, axis = -2)

r_chlb = np.linspace(0,200)
r_chla = (l/phi-r_chlb.reshape(-1,1,1)*alphas[1])/alphas[0]

resources = pig_contribution(pigs)

fig, ax = plt.subplots(5,4, figsize = (9,9), sharex = True, sharey = True)

for i in range(ax.size):
    axc = ax.flatten()[i]
    axc.scatter(*(resources[...,i]), c = time,linewidth = 0,
                  norm=LogNorm(vmin=time.min(), vmax=time.max()))

    axc.plot(r_chla[:,:,i],r_chlb)
    
plt.axis([0,250,0,250])

###############################################################################
# orthagonalize the pigments

