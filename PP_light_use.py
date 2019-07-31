"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Similar to plot_absorption_spectra.py but figure to be used in a 
powerpoint slide

Plot the absorption spectrum of all pigments involved
Plot the example absorption spectrum of two coexisting species
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from phytoplankton_communities.generate_species import gen_com, lambs
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
EF = [sum(equi*1e-7) for equi in equis]
k_spec = k_spec[...,0]/np.amax(k_spec)

def save_fig(axs = None, counter = [1]):
    print(counter[0])
    plt.savefig("PP_slides/Light_use_{}.png".format(counter[0]),
                transparent = "True")
    if not (axs is None):
        for ax in axs:
            for l in ax.lines:
                l.set_linewidth(2)
                l.set_alpha(0.3)
            
    counter[0] += 1            

plt.style.use('dark_background')  
mpl.rcParams.update({'font.size': 20, "xtick.labelsize": 14, 
                     "ytick.labelsize": 14})   
fig = plt.figure(figsize = (11,7))
plt.tight_layout()
gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[2, 1])


ax = np.ones((2,2), dtype = "object")

# initiate first axis and plot solar spectrum
ax[0,0] = plt.subplot(gs[0])
ax[0,0].set_xticks([400,460, 530, 570, 660, 700])
ax[0,0].set_xticklabels([400,"blue" ,"green" ,"yellow" ,"red" , 700])
ax[0,0].set_xlim([400,700])
ax[0,0].set_ylim([0,0.16])
ax[0,0].set_yticks([0,0.05, 0.1, 0.15])
ax[0,0].set_ylabel(r"Residual light")
ax[0,0].set_xlabel("Light colour")
ax[0,0].plot(lambs, I_in[:,0,0], "white", linewidth = 4)
save_fig()

# add first phytoplankton species and second axis
ax[1,0] = plt.subplot(gs[2])
ax[0,0].set_xticklabels(["","" ,"" ,"" ,"" , ""]) # imitate shared axes
ax[0,0].set_xlabel("")
ax[1,0].set_xlabel("Light colour")
ax[1,0].set_xticks([400,460, 530, 570, 660, 700])
ax[1,0].set_xticklabels([400,"blue" ,"green" ,"yellow" ,"red" , 700])
ax[1,0].set_xlim([400,700])
ax[1,0].set_ylabel(r"Absorption")
ax[1,0].set_ylim([0,1.1])
ax[1,0].set_yticks([0,0.5,1])
ax[1,0].plot(lambs, k_spec[:,0], linewidth = 4)
save_fig([ax[0,0]])

# change the light availability
ax[0,0].plot(lambs, I_out[0,:,0], linewidth = 4)
save_fig()

# add species richness plot
ax[0,1] = plt.subplot(gs[1])
ax[0,1].set_xticks(np.arange(1,5))
ax[0,1].set_xlabel("Trait richness")
ax[0,1].set_xlim([0.5,4.5])
ax[0,1].set_ylim([0.5,4.5])
ax[0,1].set_yticks(np.arange(1,5))
ax[0,1].set_ylabel("Species richness")
ax[0,1].plot(1,1,'o', color = "white")
save_fig()

# add EF axis
ax[1,1] = plt.subplot(gs[3])
ax[0,1].set_xticklabels(4*[""])
ax[0,1].set_xlabel("")
ax[1,1].set_xticks(np.arange(1,5))
ax[1,1].set_xlim([0.5,4.5])
ax[1,1].set_ylim([3,5])
ax[1,1].set_yticks([3,4,5])
ax[1,1].set_xlabel("Trait richness")
ax[1,1].set_ylabel("Ecosystem function")
ax[1,1].plot(1,EF[0], 'o', color = "white")
save_fig(ax.flatten())

for i in range(1,4):
    ax[1,0].plot(lambs, k_spec[:,i], linewidth = 4)
    ax[0,0].plot(lambs, I_out[i,:,0], linewidth = 4)
    ax[0,1].plot(np.arange(i+1)+1, np.arange(i+1) +1, 'o', color = "white")
    ax[1,1].plot(np.arange(i+1)+1, EF[:i+1], 'o', color = "white")
    save_fig(ax.flatten())