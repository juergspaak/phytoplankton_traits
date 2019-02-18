"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Similar to plot_absorption_spectra.py but figure to be used in a 
powerpoint slide

Plot the absorption spectrum of all pigments involved
Plot the example absorption spectrum of two coexisting species
"""

import numpy as np
import matplotlib.pyplot as plt
from phytoplankton_communities.generate_species import gen_com, pigments
from phytoplankton_communities.generate_species import pigment_names, lambs
import phytoplankton_communities.richness_computation as rc
from phytoplankton_communities.I_in_functions import sun_spectrum

# fix randomness
np.random.seed(hash(1))

I_in = 40*sun_spectrum["direct full"]
zm = 100
phi,l,k_spec,alpha,found = gen_com([0,1,14,12], 3, 100,I_ins = I_in)

equi,unfixed = rc.multispecies_equi(phi/l,k_spec, I_in)

# remove all species where we have not found equilibrium
equi = equi*(1-unfixed)

index = np.argmax(np.sum(equi>0, axis = 0))
phi,l,k_spec = phi[:,index:index+1], l[index:index+1], k_spec[...,index:index+1]
equi = equi[:,index:index+1]

equis = []
I_out = []
for i in range(1,5):
    equis.append(rc.multispecies_equi((phi/l)[:i], k_spec[:,:i], I_in)[0])
    I_out.append(I_in[:,0]*np.exp(-zm*np.sum(equis[i-1]*k_spec[:,:i], axis = 1)))
I_out = np.array(I_out)
pig_colors = ["green", "darkolivegreen", "lime", "yellowgreen",
              "purple", "brown", "cyan", "red", "orange"]

def plot_pigments(array, ax, ls = '-', lw = 2):
    for i in range(len(pig_colors)):
        if max(array[i])>0:
            ax.plot(lambs, array[i]/np.amax(array), ls,color = pig_colors[i], 
                label = i, linewidth = lw)

def save_fig(axs, counter = [1]):
    print(counter[0])
    plt.savefig("PP_slides/Light_use_{}.png".format(counter[0]),
                transparent = "True")
    for ax in axs:
        for l in ax.lines:
            l.set_linewidth(2)
            l.set_alpha(0.3)
            
    counter[0] += 1            

plt.style.use('dark_background')         
fig, ax = plt.subplots(2,1, sharex = True,  figsize = (7,7))
fs_labels = 20
fs_axis = 14
ax[1].set_xticks([400,460, 530, 570, 660, 700])
ax[1].set_xticklabels([400,"blue" ,"green" ,"yellow" ,"red" , 700])
ax[1].set_xlim([400,700])
           
ax[0].set_ylim([0,1.1])
ax[0].set_yticks([0,0.5,1])
ax[0].tick_params(axis = "both", labelsize = fs_axis)

ax[1].set_ylim([0,0.16])
ax[1].set_yticks([0,0.05, 0.1, 0.15])
ax[1].tick_params(axis = "both", labelsize = fs_axis)

ax[0].set_ylabel(r"Absorption", fontsize = fs_labels)
ax[1].set_ylabel(r"Residual light", fontsize = fs_labels)
ax[1].set_xlabel("Light colour", fontsize = fs_labels)
l0 = ax[1].plot(lambs, I_in[:,0,0], "white", linewidth = 4)[0]
save_fig(ax)

k_spec = k_spec[...,0]/np.amax(k_spec)
for i in range(4):
    l1 = ax[0].plot(lambs, k_spec[:,i], linewidth = 4)[0]
    l2 = ax[1].plot(lambs, I_out[i,:,0], linewidth = 4)[0]
    save_fig(ax)

    
"""
for l in ax.lines:
    l.set_alpha(0.3)
    l.set_linewidth(lw_old)"""

