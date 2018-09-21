"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Load absorption spectra of pigments found in nature
Load also pigments that are not phytosyntheticall active

Reference papers can be found in the Pigment_algae_table.csv
"""

import numpy as np
from scipy.integrate import simps, odeint
from generate_species import gen_com, pigments
from I_in_functions import sun_spectrum, k_BG, zm, dlam
from richness_computation import multi_growth
import matplotlib.pyplot as plt
import pigments as pig
from matplotlib.colors import LogNorm

i_lux,i_kBG = 20,20
luxs = np.linspace(20,60,i_lux)
glivin_conc = np.linspace(0,5,i_kBG)

phi,l, k_spec, alphas,a = gen_com([9,9], 1, 16)
l[:] = np.mean(l)
time = (24*100)**np.linspace(0,1)
N_all = np.empty((i_lux, i_kBG, len(time))+phi.shape)
k_BGs = np.empty((i_kBG,len(k_spec),1,1))
I_ins = np.empty((i_lux,1,len(k_spec),1))

for i in range(i_lux):
    print(i)
    for j in range(i_kBG):
        I_in = sun_spectrum["usual"]*luxs[i]
        I_in_t = lambda t: I_in
        k_BGs[j] = (k_BG["ocean"]+k_BG["peat lake"]*glivin_conc[j]).reshape(-1,1,1)
        I_ins[i,0] = I_in.reshape(-1,1)
        N0 = np.full(phi.size, 1e8)


        N_t = odeint(multi_growth, N0, time, args = (I_in_t,k_spec, phi,l,zm,
                        k_BGs[j], True))

        N_all[i,j] = N_t.reshape((-1,) +phi.shape)


# compute the contribution of both pigments
pigs = pigments[[0,6]]
pigs.shape = len(pigs),1,1,-1,1

tot_abs = zm*(0*np.nansum(N_all[:,:,[0]]*k_spec, axis = -2)
                +k_BGs[...,0])
resources = simps(pigs/tot_abs*I_ins*(1-np.exp(-tot_abs)),
                 dx = dlam, axis = -2)


r_chlb = np.linspace(0,1000)
r_chla = (l/phi-r_chlb.reshape(-1,1,1)*alphas[6])/alphas[0]

fig, ax = plt.subplots(4,4, figsize = (9,9), sharex = True, sharey = True)

for i in range(ax.size):
    axc = ax.flatten()[i]
    axc.scatter(resources[0,...,i].flatten(), resources[1,...,i].flatten(), 
                c = (N_all[...,-1,0,i]/np.sum(N_all[...,-1,:,i],axis = -1)).flatten(), 
                linewidth = 0, vmin = 0, vmax = 1, s = 10)
    axc.plot(r_chla[:,:,i],r_chlb)
plt.axis([0,None, 0,None])
###############################################################################
# orthagonalize the pigments"""
fig.savefig("idea of coexistence in competition for light.pdf")
