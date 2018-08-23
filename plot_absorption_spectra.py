"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the main text,
for more information check the description in the main text

Plot the absorption spectrum of all pigments involved
Plot the example absorption spectrum of two coexisting species
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_species import gen_com, pigments, pigment_names
import richness_computation as rc
from I_in_functions import sun_spectrum
from pigments import lambs

# fix randomness
np.random.seed(20110505)

I_in = 40*sun_spectrum["direct full"]

[phi,l],k_spec,alpha,found = gen_com([3,-5], 3, 50,I_ins = I_in)

equi,unfixed = rc.multispecies_equi(phi/l,k_spec, I_in)

# remove all species where we have not found equilibrium
equi = equi*(1-unfixed)

index = np.argmax(np.sum(equi>0, axis = 0))
pig_colors = ["green", "darkolivegreen", "lime", "yellowgreen",
              "purple", "brown", "cyan", "red", "orange"]

def plot_pigments(array, ax, ls = '-', lw = 2):
    for i in range(len(pig_colors)):
        if max(array[i])>0:
            ax.plot(lambs, array[i], ls,color = pig_colors[i], 
                label = i, linewidth = lw)

fig, ax = plt.subplots(3,1,figsize = (8,9),sharex = True)
ax[2].get_shared_y_axes().join(ax[2],ax[1])
plot_pigments(pigments, ax[0])
ax[0].legend(pigment_names, ncol = 3, fontsize = 10)

# plot the absorption spectrum of the species
ax[1].plot(lambs,10**9*(k_spec)[:,0, index], linewidth = 2, color = "black",
        label = "Prochlorophyta example")
ax[1].legend()
# add the decomposition of the absorption spectra
plot_pigments(10**9*alpha[:,0,index, np.newaxis]*pigments,ax[1], ls = '-'
              , lw = 1)

# plot the absorption spectrum of the species
ax[2].plot(lambs,10**9*(k_spec)[:,1, index], linewidth = 2, color = "black",
        label = "Dinophyta example")
ax[2].legend()
# add the decomposition of the absorption spectra
plot_pigments(10**9*alpha[:,1,index, np.newaxis]*pigments,ax[2], ls = '-'
              , lw = 1)



fs = 12
# labels and legend
ax[2].set_xlabel("Wavelength [nm]", fontsize = fs)

ax[0].set_ylabel(r"Absorptiom [$cm^{-1}mM^{-1}$]", fontsize = fs)
ax[1].set_ylabel(r"Absorption [$10^{-9}cm^2fl^{-1}$]", 
                    fontsize = fs)
ax[2].set_ylabel(r"Absorption [$10^{-9}cm^2fl^{-1}$]", 
                    fontsize = fs)

# add titles
ax[0].set_title("A", loc = "left")
ax[1].set_title("B", loc = "left")
ax[2].set_title("C", loc = "left")


fig.savefig("Figure,absorption_spectra.pdf")