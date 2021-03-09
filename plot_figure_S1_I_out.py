"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot relative yield total, complemenratiry and selection effect
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12) 

try:
    data_org
except NameError:
    from load_data import data_org, max_spec
spaak_data = data_org[data_org.r_spec_start >= max_spec]
#spaak_data = data_org[data_org.r_spec_equi != 1]
spaak_data = spaak_data[spaak_data.r_spec_equi != 1]
spaak_data["I_out, equi"] /= 40/100 # change to percentage
spaak_data["I_out, t=240"] /= 40/100

names = ["EF_equi", "EF_t=240", "I_out, equi", "I_out, t=240"]
data_n_pig = {name: [] for name in names}
r_spec = []
r_pigs = np.arange(min(spaak_data["r_pig_start"]),
                   max(spaak_data["r_pig_start"])+1)
for n in r_pigs:
    ind = spaak_data.r_pig_start == n
    for name in names:
        data_n_pig[name].append(spaak_data[name].values[ind])

data_size = {name: [] for name in
        names}
ranges, dr = np.linspace(0,3, 16, retstep = True)
for i in range(len(ranges)-1):
    ind = (spaak_data.size_cv>ranges[i]) & (spaak_data.size_cv < ranges[i]+1)
    for name in names:
        data_size[name].append(spaak_data[name][ind].values)
        
fig, ax = plt.subplots(2,2, sharex = "col", sharey = "row", figsize = (7,7))

for i,a in enumerate(ax.flatten(order = "F")):
    a.set_title("ABCDEFGH"[i])

ax[0,0].set_ylabel("Total biovolume")
ax[1,0].set_ylabel("Total absorption")
ax[1,0].set_xlabel("Initial\npigment richness")
ax[1,1].set_xlabel("Initial\nCV(size)")

c_sta = "lime"
c_coe = "green"
datas = [data_n_pig, data_size]
x = [r_pigs, ranges[1:] - dr/2]
for i in range(2):
    ax[0,i].plot(x[i], [np.median(d) for d in datas[i]["EF_equi"]], '.',
      color = c_coe)
    ax[0,i].plot(x[i], [np.median(d) for d in datas[i]["EF_t=240"]], '.',
      color = c_sta)
    
    ax[1,i].plot(x[i], [np.median(d) for d in datas[i]["I_out, equi"]], '.',
      color = c_coe)
    ax[1,i].plot(x[i], [np.median(d) for d in datas[i]["I_out, t=240"]], '.',
      color = c_sta)

fig.tight_layout()
fig.savefig("Figure_S1.pdf")