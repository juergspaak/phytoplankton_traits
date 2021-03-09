"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the main text,
for more information check the description in the main text

Plot the absorption spectrum of all pigments involved
Plot the example absorption spectrum of two coexisting species
"""

import numpy as np
import matplotlib.pyplot as plt
import phytoplankton_communities.generate_species as gs
import phytoplankton_communities.richness_computation as rc
from phytoplankton_communities.I_in_functions import sun_spectrum
from scipy.integrate import simps

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16) 

# fix randomness
np.random.seed(hash(0))

I_in = 40*sun_spectrum["direct full"]

phi,l, k_photo, k_abs, alphas, species_size, tot_abs = gs.gen_com(
        [3,8], 3, 50,I_ins = I_in)

int_abs = simps(k_photo, axis = 0, dx = gs.dlam)
alphas = alphas/int_abs*gs.tot_abs_mean
k_photo = k_photo/int_abs*gs.tot_abs_mean
k_abs = k_abs/int_abs*gs.tot_abs_mean

equi,unfixed = rc.multispecies_equi(phi/l,k_photo, k_abs, I_in)

# remove all species where we have not found equilibrium
equi = equi*(1-unfixed)

index = np.argmax(np.sum(equi>0, axis = 0))
pig_colors = ["green", "darkolivegreen", "lime", "yellowgreen",
              "purple", "magenta", "blue",
              "orange", "cyan", "red", "sandybrown",
              "darkorange", "green", "lime", "darkolivegreen"]
pig_linstyles = np.array( len(gs.pigment_order)*["--"])
pig_linstyles[gs.photo] = "-"

def plot_pigments(array, ax, ls = '-', lw = 2):
    for i in range(len(pig_colors)):
        if max(array[i])>0:
            ax.plot(gs.lambs, array[i], linestyle = pig_linstyles[i],
                    color = pig_colors[i], label = i, linewidth = lw)

fig = plt.figure(figsize = (8,6))
ax_pig = fig.add_subplot(211)
plot_pigments(gs.pigments, ax_pig)
pig_short = gs.pigment_order
pig_short[3] = "Peridinin" # Peridinin
pig_short[4] = "Fucoxanthin"
pig_short[7] = r"$\beta$-car."
pig_short[8:11] = ["PCB", "PEB", "PUB"]
ax_pig.legend(pig_short, ncol = 3, fontsize = 10)
ax_pig.set_ylim([0,0.09])
ax_pig.set_yticks([0,0.04,0.08])
ax_pig.set_xticks([400,500,600,700])



ax_ex1 = fig.add_subplot(223)
ax_ex2 = fig.add_subplot(224)
ax_ex2.get_shared_y_axes().join(ax_ex1,ax_ex2)



# plot the absorption spectrum of the species
ax_ex1.plot(gs.lambs,10**9*(k_photo)[:,0, index], linewidth = 2,color = "black",
        label = "Photosyntehtic spectrum")
ax_ex1.plot(gs.lambs,10**9*(k_abs)[:,0, index], linewidth = 2, color = "black",
        label = "Absorption spectrum", linestyle = "--")
ax_ex1.legend()
# add the decomposition of the absorption spectra
plot_pigments(10**9*alphas[:,0,index, np.newaxis]*gs.pigments,ax_ex1, ls = '-'
              , lw = 1)

# plot the absorption spectrum of the species
ax_ex2.plot(gs.lambs,10**9*(k_photo)[:,1, index], linewidth = 2, color = "black",
        label = "Photosyntehtic spectrum")
ax_ex2.plot(gs.lambs,10**9*(k_abs)[:,1, index], linewidth = 2, color = "black",
        label = "Absorption spectrum", linestyle = "--")
ax_ex2.legend()
# add the decomposition of the absorption spectra
plot_pigments(10**9*alphas[:,1,index, np.newaxis]*gs.pigments,ax_ex2, ls = '-'
              , lw = 1)


fs = 12

# labels and legend
ax_pig.set_xlabel("Wavelength [nm]", fontsize = fs)
ax_ex1.set_xlabel("Wavelength [nm]", fontsize = fs)
ax_ex2.set_xlabel("Wavelength [nm]", fontsize = fs)

ax_pig.set_xlim([400,700])
ax_ex1.set_xlim([400,700])
ax_ex2.set_xlim([400,700])

ax_pig.set_ylabel(r"Absorptiom [$m^2mg^{-1}$]", fontsize = fs)
ax_ex1.set_ylabel(r"Absorption [$10^{-9}cm^2fl^{-1}$]", 
                    fontsize = fs)

# add titles
ax_pig.set_title("A", loc = "left")
ax_ex1.set_title("B", loc = "left")
ax_ex2.set_title("C", loc = "left")
ax_ex1.set_xticks([400,500,600,700])
ax_ex2.set_xticks([400,500,600,700])
fig.tight_layout()

fig.savefig("Figure_1_absorption_spectra.pdf")