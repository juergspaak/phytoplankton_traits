# -*- coding: utf-8 -*-
"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot the main Figure of the paper
See description in manuscript for more information
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

my_cmap = plt.cm.viridis
my_cmap.set_under('w',1)

# load the dataset
try:
    data_photo
except NameError:
    data_photo_org = pd.read_csv("data/data_no_photoprotectionnew.csv")
index = np.arange(len(data_photo_org))
data_photo = data_photo_org[index%2==1]
#data_photo = pd.read_csv("data/data_no_photoprotection_old_pigments.csv")
data_photo.r_pig_start = data_photo.r_pig_start.astype(int)
    
print(data_photo.shape)
plt.figure()
r_pig = np.arange(min(data_photo.r_pig_start), max(data_photo.r_pig_start)+1)
plt.hist(data_photo.r_pig_start, bins = r_pig)

pig_range = np.arange(min(data_photo.r_pig_start),
                      max(data_photo.r_pig_start) +1)


ND_cols = ["ND_{}".format(i) for i in range(5)]
ND = data_photo[ND_cols].values

FD_cols = ["FD_{}".format(i) for i in range(5)]
FD = data_photo[FD_cols].values
FD[ND == 1] = np.nan
ND[ND == 1] = np.nan

sort_by = data_photo.r_pig_start
cases = list(set(sort_by))

ND_box = [ND[data_photo.r_pig_start == i].flatten() for i in r_pig]
ND_box = [N[np.isfinite(N)] for N in ND_box]
FD_box = [-FD[data_photo.r_pig_start == i].flatten() for i in r_pig]
FD_box = [F[np.isfinite(F)] for F in FD_box]


fig = plt.figure(figsize = (7,7))
ax = [fig.add_subplot(2,2,1), fig.add_subplot(2,2,3)]
ax[0].boxplot(ND_box, sym = "", positions = r_pig)
ax[1].boxplot(FD_box, sym = "", positions = r_pig)

ax[0].set_ylabel("$\mathcal{N}_i$")
ax[1].set_ylabel("$\mathcal{-F}_i$")
ax[1].set_xlabel("initial pigment richness")

hist_range = np.round([np.nanpercentile(ND,[1,99]),
                       np.nanpercentile(-FD, [1,99])],2)
ax_coex = fig.add_subplot(1,2,2)

# want to only show percent of the data in the histogram plot
nbins = 201 # number of bins in histogram
percents = [0.01, 0.05, 0.25, 0.50, 0.75]

counts, xedges, yedges = np.histogram2d(ND[np.isfinite(ND)],
                                           FD[np.isfinite(FD)], nbins,
                                           range = [[-0.4,0.4],[-0.4,0.4]])

ind = [np.argmin(np.cumsum(np.sort(counts.flatten()))/np.sum(counts)<percent)
                 for percent in percents]
bounds = np.sort(counts.flatten())[ind]-1

cmap = plt.get_cmap("viridis", len(ind))
counts_colored = np.full(counts.shape, np.nan)
for i,value in enumerate(np.linspace(0,1,len(ind)*2+1)[1::2]):
    counts_colored[counts>bounds[i]] = value
    
im = ax_coex.imshow(counts_colored.T, aspect = "auto", cmap = cmap,
                   extent = np.append(yedges[[0,-1]], xedges[[0,-1]]),
                   vmin = 0, vmax = 1)
cbar = fig.colorbar(im, ax = ax_coex, alpha = 1)
cbar.ax.get_yaxis().set_ticks(np.linspace(0,1,len(ind)*2+1)[1::2])
cbar.ax.get_yaxis().set_ticklabels(["{}%".format(int(100*(1-percent)))
            for percent in percents])
    
# add coex line
ND_line = np.linspace(*ax_coex.get_xlim(), 101)
ax_coex.plot(ND_line, ND_line/(1-ND_line), "red")

ax_coex.set_xlim(-0.05,0.2)
ax_coex.set_ylim(-0.2,0.1)
ax_coex.set_yticks([-0.2,-0.1,0,0.1])
ax_coex.axhline(0, color = "k")
ax_coex.axvline(0, color = "k")
ax_coex.set_xlabel(r"$\mathcal{N}_i$")
ax_coex.set_ylabel(r"$\mathcal{-F}_i$")
ax[0].set_yticks([-0.05, 0, 0.05, 0.1, 0.15])
ax[1].set_yticks([-0.1, -0.05, 0,  0.05, 0.1])
ax[0].set_title("A")
ax[1].set_title("B")
ax_coex.set_title("C")
fig.tight_layout()

fig.savefig("figure_traits_NFD_barplot.pdf")