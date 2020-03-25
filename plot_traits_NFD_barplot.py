# -*- coding: utf-8 -*-
"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot the main Figure of the paper
See description in manuscript for more information
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

my_cmap = plt.cm.viridis
my_cmap.set_under('w',1)

# load the dataset
try:
    spaak_data
except NameError:
    spaak_data = pd.read_csv("data/data_ND_all.csv")

pig_range = np.arange(min(spaak_data.r_pig_start),
                      max(spaak_data.r_pig_start) +1)


ND_cols = ["ND_{}".format(i) for i in range(1,6)]
ND = spaak_data[ND_cols].values

FD_cols = ["FD_{}".format(i) for i in range(1,6)]
FD = spaak_data[FD_cols].values
FD[ND == 1] = np.nan
ND[ND == 1] = np.nan
equi_cols = ["equi_{}".format(i) for i in range(1,6)]
equi = spaak_data[equi_cols].values

sort_by = spaak_data.r_pig_start
cases = list(set(sort_by))
color = rainbow(np.linspace(0,1,len(cases)))

ND_box = [ND[spaak_data.r_pig_start == i].flatten() for i in range(2,10)]
ND_box = [N[np.isfinite(N)] for N in ND_box]
FD_box = [-FD[spaak_data.r_pig_start == i].flatten() for i in range(2,10)]
FD_box = [F[np.isfinite(F)] for F in FD_box]


fig = plt.figure(figsize = (7,7))
ax = [fig.add_subplot(2,2,1), fig.add_subplot(2,2,3)]
ax[0].boxplot(ND_box, sym = "", positions = range(2,10))
ax[1].boxplot(FD_box, sym = "", positions = range(2,10))

ax[0].set_ylabel("$\mathcal{N}$")
ax[1].set_ylabel("$\mathcal{-F}$")
ax[1].set_xlabel("initial pigment richness")

hist_range = np.round([np.nanpercentile(ND,[5,95]),
                       np.nanpercentile(-FD, [5,95])],2)
ax_coex = fig.add_subplot(1,2,2)

# want to only show percent of the data in the histogram plot
nbins = 50 # number of bins in histogram
percent = 0.95 # percentage of data to show
counts, xedges, yedges = np.histogram2d(ND[np.isfinite(ND)],
                                           -FD[np.isfinite(FD)], nbins)
counts = np.sort(counts.flatten())[::-1]
ind = np.argmin(np.cumsum(counts)/np.sum(counts)<percent)
cmin = counts[ind]-1
res = ax_coex.hist2d(ND[np.isfinite(ND)], -FD[np.isfinite(FD)], nbins,
                        cmin = cmin)
ax_coex.set_xlim(0,0.2)
ax_coex.set_ylim(-0.25,0.15)

ax[0].set_yticks([-0.05, 0, 0.05, 0.1, 0.15])
ax[1].set_yticks([-0.1, -0.05, 0,  0.05, 0.1])
ax[0].set_title("A")
ax[1].set_title("B")
ax_coex.set_title("C")
fig.tight_layout()

fig.savefig("figure_traits_NFD_barplot.pdf")