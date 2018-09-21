"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Load absorption spectra of pigments found in nature
Load also pigments that are not phytosyntheticall active

Reference papers can be found in the Pigment_algae_table.csv
"""

import numpy as np
from scipy.integrate import simps, odeint
from generate_species import gen_com
from I_in_functions import sun_spectrum, k_BG, zm, dlam
from richness_computation import multi_growth
import matplotlib.pyplot as plt
from numerical_NFD import find_NFD

#np.random.seed(hash("hello")%4294967295)

I_in = sun_spectrum["usual"]*40
I_in_t = lambda t: I_in
phi,l, k_spec, alphas,a = gen_com([0,9], 1, 100)
l[:] = np.mean(l)

k_key = "ocean"
N0 = np.full(phi.size, 1e6)
time = (24*100)**np.linspace(0,1,50)

N_t = odeint(multi_growth, N0, time, args = (I_in_t,k_spec, phi,l,zm,
                        k_BG[k_key].reshape(-1,1,1), True))

N_t.shape = (len(time),) + phi.shape

i = np.argmax((N_t[-1]>np.sum(N_t[-1],axis = 0)/100).all(axis = 0))

NO_t = np.empty((len(time),2))
FD_t = np.empty((len(time),2))
c_t = np.empty(len(time))

def per_capita(N):
    tot_abs = zm*(np.sum(k_spec[...,i]*N,axis = 1)+k_BG[k_key])[:,None]
    return phi[:,i]*simps(k_spec[...,i]/tot_abs*(1-np.exp(-tot_abs))*I_in[:,None],
            axis = 0,dx = dlam)-l[i]

for t in range(len(time)):
    pars = find_NFD(per_capita, pars = {"N_star": np.ones((2,2))*N_t[t,:,i]},
                                        force = True)
    NO_t[t] = pars["NO"]
    FD_t[t] = pars["FD"]
    c_t[t] = pars["c"][0,1]


fig,ax = plt.subplots(2,2, figsize = (9,9), sharex = True)

# plot results from invasion
pars = find_NFD(per_capita, pars = {"N_star": np.ones((2,2))*N_t[t,:,i]})

ax[0,0].plot(time, N_t[...,i])
ax[0,1].plot(time, NO_t)
ax[1,0].plot(time, FD_t)
ax[1,1].plot(time, c_t)

ax[1,0].set_xlabel("Time")
ax[1,1].set_xlabel("Time")

ax[0,0].set_ylabel("Species density")
ax[0,1].set_ylabel("NO")
ax[1,0].set_ylabel("FD")
ax[1,1].set_ylabel("c")

