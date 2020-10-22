"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the appendix,
for more information check the description in the appendix

Plot the effects of pigment richness on Base productivity and biovolume
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
spaak = pd.read_csv("data/data_EF_time_all.csv")

# convert data from cells to fl/ml
for col in spaak.columns:
    if col[:2] == "EF":
        spaak[col] *=1e-9

def medians(x_val, y_val):
    # compute averages
    x_val, y_val = spaak[x_val], spaak[y_val]
    x_range = np.arange(min(x_val), max(x_val)+1)
    return x_range, np.array([np.nanmedian(y_val[x_val==x]) for x in x_range])

# range of pigments                       
pig_range = np.arange(min(spaak["r_pig_start"]), max(spaak["r_pig_start"]))



# plot boxes
def boxs(x_val, y_val, x_range,ax,color, position = None):
    # compute averages
    x_val, y_val = spaak[x_val], spaak[y_val]
    def_col = dict(color= color)
    ax.boxplot([y_val[x_val==x] for x in x_range], boxprops = def_col,
               whiskerprops = def_col, capprops = def_col,
               medianprops = def_col, showfliers = False, positions = position)
time = "equi"
fig, ax = plt.subplots(figsize = (7,7))
axt = ax.twinx() 
boxs("r_pig_start", "EF_"+time,pig_range,ax, 'lime',position = pig_range-0.2)
boxs("r_pig_start", "base_prod, "+time, pig_range, axt, "red")

ax.set_xlabel("Pigment richness")
ax.set_ylabel(r"Biovolume $[fl\,ml^{-1}]$")
axt.set_ylabel(r"Base productivity ($\phi\cdot l^{-1}$)")

fig.savefig("Figure,ap_sampling_effect.pdf")