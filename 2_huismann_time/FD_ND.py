"""
this file computes the niche and fitness differences of the species
"""

import communities_analytical as com
import r_i_analytical_continuous as ana
import numpy as np
import matplotlib.pyplot as plt

species = com.gen_species(1000)
lights = 75
equi = com.equilibrium(species, lights)

def f(N,species = species,I_in = lights):
    k,H,p,l = species
    I_out = I_in*np.exp(-np.sum(k*N,axis = 1))
    print(I_out.shape)
    return 1/(np.sum(k*N,axis = 1))*p*np.log((H+I_in)/(H+I_out))-l

i,r = [0,1],[1,0]

N_fn = np.zeros((2,) + equi.shape)
N_fn[r,r] = equi

N_r_i = np.zeros((2,) + equi.shape)
N_r_i[r,i] = equi

r_i2 = ana.constant_I_r_i(species, lights)

r_i = f(N_r_i)


f_N = f(N_fn)
k,H,p,l = species
f_0 = p*lights/(lights+H)-l

ND = (r_i-f_N)/(f_0-f_N)
FD = -f_N/f_0

plt.scatter(ND[0],FD[0], c = (r_i/f_0)[0], linewidths = 0, s = 4,
            vmin = -0.1, vmax = 0.1)
plt.colorbar()
plt.axis([-10,1,-1,5])

plt.show()
print(r_i2[...,:5],"\n", r_i[...,:5])