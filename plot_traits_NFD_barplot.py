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
    data_ND
except NameError:
    data_photo = pd.read_csv("data/data_no_photoprotection_old_pigments")
    data_photo.r_pig_start = data_photo.r_pig_start.astype(int)
    # only use data of communities with more than one species
    data_ND = data_photo[data_photo.r_spec_equi != 1]
    data_ND = data_photo[data_photo.r_spec_equi == 2]
    
###############################################################################
# data analysis of the dataframe
print(data_ND.shape)
plt.figure()
r_pig = np.arange(min(data_ND.r_pig_start), max(data_ND.r_pig_start)+1)
hist = plt.hist(data_ND.r_pig_start, bins = r_pig-0.5)
plt.xlabel("pigments richness")
plt.ylabel("number of communities")
###############################################################################

r_pig = np.arange(min(data_ND.r_pig_start), max(data_ND.r_pig_start)+1)
pig_range = np.arange(min(data_ND.r_pig_start),
                      max(data_ND.r_pig_start) +1)


ND_cols = ["ND_{}".format(i) for i in range(5)]
ND = data_ND[ND_cols].values

FD_cols = ["FD_{}".format(i) for i in range(5)]
FD = data_ND[FD_cols].values

ND = ND+FD-ND*FD


sort_by = data_ND.r_pig_start
cases = list(set(sort_by))

ND_box = [ND[data_ND.r_pig_start == i].flatten() for i in r_pig]
ND_box = [N[np.isfinite(N)] for N in ND_box]
FD_box = [-FD[data_ND.r_pig_start == i].flatten() for i in r_pig]
FD_box = [F[np.isfinite(F)] for F in FD_box]

###############################################################################
# plotting results
fig, ax = plt.subplots(2,2,figsize = (7,7), sharey = "row", sharex = "col")
ax[0,0].boxplot(ND_box, sym = "", positions = r_pig)
ax[1,0].boxplot(FD_box, sym = "", positions = r_pig)

ax[0,0].set_ylabel("$\mathcal{N}_i$")
ax[1,0].set_ylabel("$\mathcal{-F}_i$")
ax[1,0].set_xlabel("initial\npigment richness")

###############################################################################
# plot of sive variation
x = "size_cv"


x_dat = data_ND[x].values
ranges,dr = np.linspace(*np.percentile(x_dat, [1,99]), 15,
                        retstep = True)

ND_cols = ["ND_{}".format(i) for i in range(5)]
FD_cols = ["FD_{}".format(i) for i in range(5)]
ND_box = []
FD_box = []
EF_equi = []
EF_mid = []
for i in range(len(ranges)-1):
    ind = (x_dat>ranges[i]) & (x_dat<ranges[i+1])
    ND_box.append(data_ND[ND_cols][ind].values)
    ND_box[-1] = ND_box[-1][np.isfinite(ND_box[-1])]
    FD_box.append(data_ND[FD_cols][ind].values)
    FD_box[-1] = FD_box[-1][np.isfinite(FD_box[-1])]
    
    EF_equi.append(data_ND["EF_equi"][ind].values)
    EF_mid.append(data_ND["EF_t=240"][ind].values)
    
ax[0,1].boxplot(ND_box, positions = ranges[1:] - dr/2, sym = "",
  widths = 0.8*dr)

ax[1,1].boxplot(FD_box, positions = ranges[1:] - dr/2, sym = "",
  widths = 0.8*dr)
ax[1,1].set_xlabel("Initial\nCV(size)")

ax[0,0].set_title("A")
ax[0,1].set_title("B")
ax[1,0].set_title("C")
ax[1,1].set_title("D")

ax[1,0].set_yticks([-0.2,0,0.2])
ax[0,0].set_yticks([0,0.2,0.4])
ax[1,0].set_xticks([4,5,10,r_pig[-1]])
ax[1,0].set_xticklabels([4,5,10,r_pig[-1]])

ticks = [1,2,3,4,5]
ax[1,1].set_xticks(ticks)
ax[1,1].set_xticklabels(ticks)
ax[1,1].set_xlim(ranges[[0,-1]])

fig.savefig("figure_traits_NFD_barplot.pdf")