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
np.random.seed(hash(2))

I_in = 40*sun_spectrum["direct full"]
zm = 100
phi,l,k_spec,k_abs,alpha,size, t_abs = gen_com([0,1,14,5], 3, 100,
                                               I_ins = I_in)
phi = phi/2.5
phi[-1]*=1.5
k_abs = k_spec
equi,unfixed = rc.multispecies_equi(phi/l,k_spec,k_abs, I_in)

# remove all species where we have not found equilibrium
equi = equi*(1-unfixed)

index = np.argmax(np.sum(equi>0, axis = 0))
phi,l,k_spec = phi[:,index:index+1], l[index:index+1], k_spec[...,index:index+1]
k_abs = k_spec
equi = equi[:,index:index+1]

equis = []
I_out = []
for i in range(1,5):
    equis.append(rc.multispecies_equi((phi/l)[:i], k_spec[:,:i],k_abs[:,:i], I_in)[0])
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

ax_r = fig.add_subplot(1,3,3)
ax_r.set_xticks(np.arange(1,5))
ax_r.set_yticks(np.arange(1,5))
ax_r.set_xlabel("species richness")
ax_r.set_ylabel("Trait richness")
ax_r.set_xlim([0.9,4.1])
ax_r.set_ylim([0.9,4.1])
ax_all = [ax[0,0],ax[1,0],ax_r]


for i in range(1,4):
    ax[1,0].plot(lambs, k_spec[:,i], linewidth = 4)
    ax[0,0].plot(lambs, I_out[i,:,0], linewidth = 4)
    ax_r.plot(np.arange(i+1)+1, np.arange(i+1) +1, 'o', color = "white")
    save_fig(ax_all)
    
print(equis[-1])