# -*- coding: utf-8 -*-
"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the appendix,
for more information check the description in the appendix
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16) 

# load the dataset
spaak_data = pd.read_csv("data/data_ap_richness_all.csv")

# convert to integer
spaak_data.r_pig_start = spaak_data.r_pig_start.astype(int)                                      

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
    x_spe[x_spe<1e-3] = np.nan # raised warning concerns prexisting nan
    # minimal value is 1 and not 0 as usually in plt.imshow
    extent = x_lim+[0.5,5.5]
    # dont plot the cases with ss = 0 or se = 0
    im = ax.imshow(x_spe[1:,1:], interpolation = "nearest",extent = extent,
        origin = "lower",aspect = "auto",vmax = 1,vmin = 0)
    
    # add averages to the figures
    av_spe = np.nansum(x_spe*np.arange(6)[:,np.newaxis],axis = 0)
    ax.plot(np.arange(len(av_spe)),av_spe,'o', color = "blue")
    return im

# actual plotting                                      
fig, ax = plt.subplots(2,1,sharex = True, sharey = True, figsize = (9,4))    
    
for i,case in enumerate(["Const1", "Fluctuating"]):

    # divide into fluctuating and constant case
    data = spaak_data[spaak_data.case == case]
    
    diff = np.array([-0.5,0.5])
    ax0_Xlim = [0.5,9.5]

    # plot probability distributions
    im = plot_imshow(data, "r_pig_start", ax[i],ax0_Xlim)
    
    # change axis
    # number of species typically within 1 and 5
    plt.ylim([0.5,5.5])
    
    plt.xlim(ax0_Xlim)
    plt.xticks([1,5,9])
    

    
    plt.xlabel("Initial pigment richness")
    print(case + " communities", np.sum(data["n_fix"]))
    
# set axis labels
ax[0].set_ylabel("Species\nrichness")
ax[1].set_ylabel("Species\nrichness")
ax[0].set_title("A: Constant light")
ax[1].set_title("B: Flutuating light")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig("Figure_ap_diversity.pdf")