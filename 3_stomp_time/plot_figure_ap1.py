"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Explain the increase in EF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

spaak_data = pd.read_csv("data/data_EF_time[3.5, 2].csv")

for col in spaak_data.columns:
    if col[:2] == "EF":
        spaak_data[col] *=1e-9

spaak_data["EF, adjust"] = spaak_data["EF, equi"]/spaak_data["fit_equi"]
spaak_data["EF, adjust2"] = spaak_data["EF, t=120"]/spaak_data["fit_start"]
                     
# plot boxes
def boxs(x_val, y_val, x_range,ax,color):
    # compute averages
    ax.set_xlabel(x_val)
    ax.set_ylabel(y_val)
    
    x_val, y_val = spaak_data[x_val], spaak_data[y_val]
    def_col = dict(color= color)
    ax.boxplot([y_val[x_val==x] for x in x_range], boxprops = def_col,
               whiskerprops = def_col, capprops = def_col,
               medianprops = def_col, showfliers = False)

    
fig, ax = plt.subplots(6,1,figsize = (9,9))
plt.subplots_adjust(hspace=0.5)

boxs("r_pig, start", "r_pig, equi", range(1,23), ax[0],"green")

boxs("r_pig, start", "fit_equi", range(1,23), ax[1],"green")

boxs("r_pig, start", "EF, equi", range(1,23), ax[2],"green")

x,y = "r_pig, equi", "EF, adjust"
ax[3].set_xlabel(x)
ax[3].set_ylabel(y)
x,y = spaak_data[x], spaak_data[y]
ax[3].scatter(x,y,c = spaak_data["r_pig, start"], linewidth = 0)

m,q,a,b,c = linregress(x,y)
ax[3].plot(np.percentile(x,[1,99]),np.percentile(x,[1,99])*m+q)

boxs("r_spec", "EF, equi", range(1,23), ax[4],"green")

boxs("r_spec", "EF, adjust", range(1,23), ax[5],"green")

