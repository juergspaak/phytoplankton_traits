# -*- coding: utf-8 -*-
"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot the main Figure of the paper
See description in manuscript for more information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from itertools import permutations

# load the dataset
spaak_data = pd.read_csv("data/data_richness_all.csv")
spaak_data = spaak_data[spaak_data.case != "Fluctuating"]
#spaak_data = spaak_data[spaak_data.r_pig_start == 3]

pig_range = np.arange(min(spaak_data.r_pig_start),
                      max(spaak_data.r_pig_start) +1)

# convert to integer
spaak_data.r_pig_start = spaak_data.r_pig_start.astype(int)
spec_range = np.arange(2,6) # don't include species richness 1 for NFD 
weights = spaak_data[["spec_rich,{}".format(i) for i in spec_range]].values
spaak_data["ND_av"] = np.nanmean(spaak_data[["ND,{}".format(n)
    for n in spec_range]] * weights, axis = 1)/np.nansum(weights, axis = 1)
spaak_data["FD_av"] = np.nanmean(spaak_data[["FD,{}".format(n)
    for n in spec_range]] * weights, axis = 1)/np.nansum(weights, axis = 1)
spaak_data = spaak_data[np.isfinite(spaak_data.ND_av)]

fig = plt.figure()
ax = plt.gca()
EF = spaak_data["biovolume,50"]
cmap = ax.scatter(spaak_data.ND_av, spaak_data.FD_av, c = EF,
            s = 4,norm=matplotlib.colors.LogNorm(), vmin = np.percentile(EF, 5)
            , vmax = np.percentile(EF, 95))
cbar = fig.colorbar(cmap)
percent = 5

ax.set_xlim([0,np.nanpercentile(spaak_data.ND_av,95)])
ax.set_ylim(np.nanpercentile(spaak_data.FD_av, [5,95]))
ax.set_xticks([0,0.02, 0.04])
ax.set_yticks([0, 0.005, 0.01, 0.015])
ax.set_xlabel("Niche differences", fontsize = 14)
ax.set_ylabel("Fitness differences", fontsize = 14)
cbar.set_label(r"Biovolume $[fl\,ml^{-1}]$", fontsize = 14)

spaak_data["ND*FD"] = spaak_data.ND_av*spaak_data.FD_av
spaak_data["r_i"] = spaak_data.ND_av + spaak_data.FD_av -spaak_data["ND*FD"]
spaak_data.rename(columns = {"ND_av":"ND", "FD_av":"FD"}, inplace = True)
spaak_data["ND2"] = np.abs(spaak_data.ND)**0.5
spaak_data["FD2"] = np.abs(spaak_data.FD)**0.5

pred_var = np.array(["ND", "FD", "ND2", "FD2", "ND*FD"])
df = pd.DataFrame(None, columns = ["intercept", "R2"] + list(pred_var))

cases = np.zeros((2**len(pred_var), len(pred_var)), dtype = bool)
id_case = np.arange(2**len(pred_var))
for i in range(len(pred_var)):
    cases[(id_case//(2**i))%2 == 0,i] = True
cases = ~cases[1:]
for i,case in enumerate(cases):
    reg = LinearRegression()
    reg.fit(spaak_data[pred_var[case]], np.log(EF))
    R2 = r2_score(np.log(EF), reg.predict(spaak_data[pred_var[case]]))
    df.loc[i, "R2"] = R2
    df.loc[i, pred_var[case]] = reg.coef_
    df.loc[i, "intercept"] = reg.intercept_
    
print(df.iloc[np.argsort(df.R2)])
fig.savefig("Figure, NFD_EF.pdf")