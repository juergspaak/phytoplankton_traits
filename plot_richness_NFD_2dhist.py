

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

sort_by = spaak_data.r_spec_equi

fig, ax = plt.subplots(3,1, sharex = True, sharey = True, figsize = (9,9))
for i, current in enumerate([2,3,4]):

    ax_c = ax.flatten()[i]
    ax_c.set_title(current)
    index = sort_by == current
    ax_c.hist2d(ND[index].flatten(), -FD[index].flatten(), bins = 50,
               range = [[0, 0.1],[-0.05,0.09]],
               normed = True, cmin = 1)

fig.savefig("figure_richness_NFD_hist2d.pdf")