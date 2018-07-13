# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
spaak_data = pd.read_csv("data/data_appendix_all.csv")
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
    print(max_x)
    # array to contain information about probabilities
    x_spe = np.zeros((6,max_x))
    for ss in range(max_x):
        # choose all cases, where starting richness is `ss`
        x = spaak_data[spaak_data[x_val]==ss]
        for se in range(6):
            x_spe[se,ss] = np.mean(x["spec_rich,{}".format(se)])
    
    # for better contrast
    x_spe[x_spe<1e-3] = np.nan
    # minimal value is 1 and not 0 as usually in plt.imshow
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

diff = np.array([-0.5,0.5])
ax0_Xlim = list(diff + np.percentile(spaak_data["r_pig_start"],[0,100]))
ax_0Xlim = list(diff + np.percentile(spaak_data["species_diversity"],[0,100]))

# plot probability distributions
plot_imshow(const_data, "species_diversity", ax[0,1],ax_0Xlim)
plot_imshow(const_data, "r_pig_start", ax[0,0],ax0_Xlim)
plot_imshow(fluct_data, "species_diversity", ax[1,1],ax_0Xlim)
im = plot_imshow(fluct_data, "r_pig_start", ax[1,0],ax0_Xlim)

# change axis
# number of species typically within 1 and 5
ax[0,0].set_ylim([0.5,5.5])

ax[0,0].set_xlim(ax0_Xlim)
ax[0,0].set_xticks([1,5,10])

ax[1,1].set_xlim(ax_0Xlim)
ax[1,1].set_xticks([1,5,10,15])

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

fig.savefig("Figure,appendix_diversity.pdf")