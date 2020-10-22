"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot relative yield total, complemenratiry and selection effect
"""

import matplotlib.pyplot as plt
import pandas as pd

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12) 

spaak_data = pd.read_csv("data/data_EF_time_all.csv")
#spaak_data = data.copy()

comp, selec, ryt, EF, EF2 = [],[],[], [], []
r_spec = []
for i,n in enumerate(sorted(set(spaak_data["r_pig_start"]))):
    ind = spaak_data["r_pig_start"] == n
    comp.append(spaak_data.complementarity[ind])
    selec.append(spaak_data.selection[ind])
    ryt.append(spaak_data.RYT[ind])
    EF.append(spaak_data["EF_equi"][ind])
    EF2.append(spaak_data["EF_t=240"][ind])
    r_spec.append(spaak_data["r_spec_start"][ind])

fig, ax = plt.subplots(2,2, figsize = (9,9), sharex = True)

ax[0,0].boxplot(EF, showfliers = False)

ax[0,0].set_ylabel("Ecosystem function")

ax[1,1].boxplot(ryt, showfliers = False)
ax[1,1].set_ylabel("Relative yield total")

ax[0,1].boxplot(comp, showfliers = False)
ax[0,1].set_ylabel("Complementarity")

ax[1,0].boxplot(selec, showfliers = False)
ax[1,0].set_ylabel("Selection")
ax[0,1].set_ylim(ax[1,0].get_ylim())

ax[1,1].set_xlabel("initial pigment richness")
ax[1,0].set_xlabel("initial pigment richness")

ax[0,0].set_title("A")
ax[0,1].set_title("B")
ax[1,0].set_title("C")
ax[1,1].set_title("D")

fig.tight_layout()
fig.savefig("Figure_ap_comp_selec2.pdf")

print([len(e) for e in EF])
plt.figure()
plt.boxplot(r_spec, showfliers=False)