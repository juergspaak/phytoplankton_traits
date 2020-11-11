# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# load the dataset
spaak_data = pd.read_csv("data/data_richness_all.csv")
# add species diversity                                     
spaak_data["species_diversity"] = [len(spec[1:-1].split()) 
                                        for spec in spaak_data.species]

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
    return im, [np.arange(len(av_spe)), av_spe]


plt.style.use('dark_background')
# actual plotting                                      
fig = plt.figure(figsize = (9,4))
# divide into fluctuating and constant case
data = spaak_data[spaak_data.case == "Const1"]

diff = np.array([-0.5,0.5])
ax0_Xlim = list(diff + np.percentile(spaak_data["r_pig_start"],[0,100]))
ax_0Xlim = list(diff + np.percentile(spaak_data["species_diversity"],[0,100]))

# plot probability distributions
im, averages = plot_imshow(data, "r_pig_start", plt.gca(),ax0_Xlim)


# change axis
# number of species typically within 1 and 5
plt.ylim([0.5,5.5])
plt.xlim(ax0_Xlim)

fs_axis = 14
fs_label = 20
plt.xticks([1,5,9], fontsize = fs_axis)
plt.yticks(range(1,6), fontsize = fs_axis)

# set axis labels
plt.ylabel("Species richness", fontsize = fs_label)

plt.xlabel("Pigment richness", fontsize = fs_label)
fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
colorbar = fig.colorbar(im, cax=cbar_ax)
colorbar.set_ticks([0,0.5,1])
colorbar.ax.tick_params(labelsize = fs_axis)
colorbar.set_label("Probability", fontsize = fs_label-2)

ax_main = fig.axes[0]

plt.gcf().subplots_adjust(bottom=0.2) # make space for xlabel
pp_col = np.array([63,63,63])/256 # background color of powerpoint

for i,xlim in enumerate([4.5,3.5,2.5,1.5,0.5]):
    rect = Rectangle((xlim,0.5),9.5,5.5, color = pp_col, fill = True)
    ax_main.add_patch(rect)
    if xlim != 0.5:
        ax_main.set_xticks([1,5,9,xlim-0.5])
    fig.savefig("PP_slides/PP, trait species diversity_{}.png".format(5-i),
                transparent = "True")
    rect.remove()
    
    
fig.savefig("PP_slides/PP, trait species diversity_6.png",
                transparent = "True")
ax_main.plot(averages[0], averages[1], 'bo')

fig.savefig("PP_slides/PP, trait species diversity_7.png",
                transparent = "True")













