"""@author: J.W: Spaak, jurg.spaak@unamur.be

Script to show that omitting back ground absorption does not change results
"""

import numpy as np
import communities_analytical as com
from real_I_out import real_I_out_r_i

import matplotlib.pyplot as plt

n = 10000
species = com.gen_species(n)
k,H,p,l = species

# I_in over time
# incoming light is sinus shaped
period = 10
size = 40
I = lambda t: size*np.sin(t/period*2*np.pi)+125 #sinus
I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period

backgrounds = np.linspace(0,10,11)
backgrounds[0] = 0.001
inv_gr = np.empty((len(backgrounds), n))

for i, background in enumerate(backgrounds):
    print(background)
    # compute the invasions growth rates for different backgrounds
    inv_gr[i] =np.amin(real_I_out_r_i(species, I, period, k_back = background),
                        axis = 0)/period

plt.figure()
plt.plot(backgrounds, np.sum(inv_gr>0,axis = 1)/n,'o')
plt.xlabe("Amount of background absorption")
plt.ylabel("Percentages of species with positive invasion growth rate")
