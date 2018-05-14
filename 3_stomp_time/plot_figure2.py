# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
spaak_data = pd.read_csv("data/data_EF_fluct_all.csv")
# add phylum diversity and species diversity at the beginning
spaak_data["phylum_diversity"] = [len(set(spec[1:-1].split())) 
                                        for spec in spaak_data.species]
                                        
spaak_data["species_diversity"] = [len(spec[1:-1].split()) 
                                        for spec in spaak_data.species]

# convert to integer
spaak_data.r_pig_start = spaak_data.r_pig_start.astype(int)                                      

# actual plotting                                      
fig,ax = plt.subplots(2,2, figsize = (9,9), sharey = True, sharex = "col")

def plot_imshow(data, x_val, ax,x_lim):
    
    # maximum value of x
    max_x = max(spaak_data[x_val])+1
    # array to contain information about probabilities
    x_spe = np.zeros((6,max_x))
    for ss in range(max_x):
        # choose all cases, where starting richness is `ss`
        x = spaak_data[spaak_data[x_val]==ss]
        for se in range(6):
            x_spe[se,ss] = np.mean(x["spec_rich,{}".format(se)])
    
    # for better contrast
    x_spe[x_spe<1e-3] = np.nan
    extent = x_lim+[0.5,5.5]
    # dont plot the cases with ss = 0 or se = 0
    im = ax.imshow(x_spe[1:,1:], interpolation = "nearest",extent = extent,
        origin = "lower",aspect = "auto",vmax = 1,vmin = 0)
    
    # add averages to the figures
    av_spe = np.nansum(x_spe*np.arange(6)[:,np.newaxis],axis = 0)
    ax.plot(np.arange(len(av_spe)),av_spe,'o')
    return im

# divide into fluctuating and constant case
fluct_data = spaak_data[spaak_data.case == "Fluctuating"]
const_data = spaak_data[spaak_data.case == "Const1"]

# plot probability distributions
plot_imshow(const_data, "species_diversity", ax[0,1],[0.5,14.5])
plot_imshow(const_data, "r_pig_start", ax[0,0],[0.5,23.5])
plot_imshow(fluct_data, "species_diversity", ax[1,1],[0.5,14.5])
im = plot_imshow(fluct_data, "r_pig_start", ax[1,0],[0.5,23.5])

# change axis
ax[0,0].set_ylim([0.5,5.5])
ax[0,0].set_xlim([1.5,23.5])
ax[0,0].set_xticks([2,5,10,15,20,23])

ax[1,1].set_xlim([0.5,14.5])
ax[1,1].set_xticks([1,5,10,14])

# set axis labels
ax[0,0].set_ylabel("Final species richness")
ax[1,0].set_ylabel("Final species richness")

ax[1,0].set_xlabel("Initial pigment richness")
ax[1,1].set_xlabel("Initial species richness")

ax[0,0].set_title("A")
ax[0,1].set_title("B")
ax[1,0].set_title("C")
ax[1,1].set_title("D")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

fig.savefig("Figure, trait species diversity.pdf")