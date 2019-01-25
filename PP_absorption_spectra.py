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
np.random.seed(hash(0))

I_in = 40*sun_spectrum["direct full"]

phi,l,k_spec,alpha,found = gen_com([12,14], 3, 50,I_ins = I_in)

equi,unfixed = rc.multispecies_equi(phi/l,k_spec, I_in)

# remove all species where we have not found equilibrium
equi = equi*(1-unfixed)

index = np.argmax(np.sum(equi>0, axis = 0))
pig_colors = ["green", "darkolivegreen", "lime", "yellowgreen",
              "purple", "brown", "cyan", "red", "orange"]

def plot_pigments(array, ax, ls = '-', lw = 2):
    for i in range(len(pig_colors)):
        if max(array[i])>0:
            ax.plot(lambs, array[i]/np.amax(array), ls,color = pig_colors[i], 
                label = i, linewidth = lw)
            

plt.style.use('dark_background')
fig = plt.figure(figsize = (8,6))
plt.gcf().subplots_adjust(bottom=0.1) #☻ make space for xlabel
ax_pig = fig.add_subplot(211)
plot_pigments(pigments, ax_pig)
ax_pig.legend(["Chl a", "Chl b", "Chl c", r"$\beta$-Car.", "Per.", "Fuc.",
               "PCB", "PEB", "PUB"], ncol = 3, fontsize = 15)




fs_labels = 20
fs_axis = 14
# labels and legend



ax_pig.set_xlim([400,700])
ax_pig.set_ylim([0,1.1])
ax_pig.set_yticks([0,0.5,1])
ax_pig.set_yticklabels([0,0.5,1],fontsize = fs_axis)
ax_pig.set_xticks([400,460, 530, 570, 660, 700])
ax_pig.set_xticklabels([400,"blue" ,"green" ,"yellow" ,"red" , 700],
                       fontsize = fs_axis)

ax_pig.set_ylabel(r"Absorption", fontsize = fs_labels)
ax_pig.set_xlabel("Light colour", fontsize = fs_labels)
fig.savefig("PP_slides/PP, absorption_spectra_1.png")

ax_ex1 = fig.add_subplot(212)
ax_pig.set_xticklabels([400,"" ,"" ,"" ,"" , 700], fontsize = fs_axis)
# plot the absorption spectrum of the species
ax_ex1.plot(lambs,k_spec[:,1, index]/np.amax(k_spec[:,1,index]),
            linewidth = 2, color = "black")
# add the decomposition of the absorption spectra
plot_pigments(alpha[:,1,index, np.newaxis]*pigments,ax_ex1, ls = '-'
              , lw = 1)

ax_pig.set_ylim([0,None])
ax_ex1.set_ylim([0,None])

ax_ex1.set_ylabel(r"Absorptiom", fontsize = fs_labels)

ax_ex1.set_xlim([400,700])
ax_ex1.set_ylim([0,1.1])
ax_ex1.set_yticks([0,0.5,1])
ax_ex1.set_yticklabels([0,0.5,1],fontsize = fs_axis)
ax_ex1.set_xticks([400,460, 530, 570, 660, 700])
ax_ex1.set_xticklabels([400,"blue" ,"green" ,"yellow" ,"red" , 700],
                       fontsize = fs_axis)

ax_pig.set_xlabel("", fontsize = fs_labels)
ax_ex1.set_xlabel("Light colour", fontsize = fs_labels)

fig.savefig("PP_slides/PP, absorption_spectra_2.png")
