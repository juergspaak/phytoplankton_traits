"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot relative yield total, complemenratiry and selection effect
"""

import matplotlib.pyplot as plt
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

comp, selec, ryt, EF = [],[],[], []
r_spec = []
for i,n in enumerate(sorted(set(spaak_data["r_pig_start"]))):
    ind = spaak_data["r_pig_start"] == n
    comp.append(spaak_data.complementarity[ind])
    selec.append(spaak_data.selection[ind])
    ryt.append(spaak_data.RYT[ind])
    EF.append(spaak_data["EF_equi"][ind])

fig, ax = plt.subplots(4,2, figsize = (9,9), sharex = "col", sharey = "row")

ax[0,0].boxplot(EF, showfliers = False)

ax[0,0].set_ylabel("Ecosystem function")

ax[3,0].boxplot(ryt, showfliers = False, whis = [5,95])
ax[3,0].set_ylabel("Relative yield total")

ax[1,0].boxplot(comp, showfliers = False, whis = [5,95])
ax[1,0].set_ylabel("Complementarity")

ax[2,0].boxplot(selec, showfliers = False, whis = [5,95])
ax[2,0].set_ylabel("Selection")
#ax[0,1].set_ylim(ax[1,0].get_ylim())

ax[3,0].set_xlabel("initial pigment richness")

for i,a in enumerate(ax.flatten(order = "F")):
    a.set_title("ABCDEFGH"[i])

names = ["EF_equi", "complementarity", "selection", "RYT"]    
data = {name: [] for name in
        names}
ranges, dr = np.linspace(0,3, 16, retstep = True)
for i in range(len(ranges)-1):
    ind = (spaak_data.size_cv>ranges[i]) & (spaak_data.size_cv < ranges[i]+1)
    for name in names:
        data[name].append(spaak_data[name][ind].values)
        
for i, name in enumerate(names):
    ax[i,1].boxplot(data[name], showfliers = False, whis = [5,95],
      positions = ranges[1:] - dr/2, widths = 0.8*dr)

ax[-1,1].set_xlabel("Initial\nCV(size)")
xticks = [0,1,2,3]
ax[-1,1].set_xticks(xticks)
ax[-1,1].set_xticklabels(xticks)
fig.tight_layout()
fig.savefig("Figure_S2_complementarity.pdf")