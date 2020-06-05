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

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16) 

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


pos = range(2,5)
ND_box = [ND[spaak_data.r_spec_equi == i].flatten() for i in pos]
ND_box = [N[np.isfinite(N)] for N in ND_box]
FD_box = [-FD[spaak_data.r_spec_equi == i].flatten() for i in pos]
FD_box = [F[np.isfinite(F)] for F in FD_box]

fig, ax = plt.subplots(2,1, sharex = True)
ax[0].boxplot(ND_box, sym = "", positions = pos)
ax[1].boxplot(FD_box, sym = "", positions = pos)

ax[0].set_ylabel("$\mathcal{N}$")
ax[1].set_ylabel("$\mathcal{-F}$")
ax[1].set_xlabel("species richness")

ax[0].set_title("A")
ax[1].set_title("B")

fig.savefig("figure_richness_NFD_barplot.pdf")