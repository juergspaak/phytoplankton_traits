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
spaak_data_pre = pd.read_csv("data/data_ND_all.csv")
#spaak_data_pre = spaak_data_pre[spaak_data_pre.r_spec_equi == 2]
ND = spaak_data_pre[["ND_{}".format(i) for i in range(1,6)]].values.flatten()
FD = spaak_data_pre[["FD_{}".format(i) for i in range(1,6)]].values.flatten()
EF = np.tile(spaak_data_pre.EF,5)
#EF = spaak_data_pre[["equi_{}".format(i) for i in range(1,6)]].values.flatten()
index = np.logical_and(np.isfinite(ND*FD),ND!=1)
ND = ND[index]
FD = FD[index]
EF = EF[index]

fig = plt.figure()
ax = plt.gca()
cmap = ax.scatter(ND, FD, c = EF,
            s = 4,norm=matplotlib.colors.LogNorm(), vmin = np.percentile(EF, 5)
            , vmax = np.percentile(EF, 95))
cbar = fig.colorbar(cmap)
percent = 1

ax.set_xlim(np.nanpercentile(ND,[percent,100-percent]))
ax.set_ylim(np.nanpercentile(FD, [percent,100-percent]))
ax.set_xlabel("Niche differences", fontsize = 14)
ax.set_ylabel("Fitness differences", fontsize = 14)
ax.invert_yaxis()
cbar.set_label(r"Biovolume $[fl\,ml^{-1}]$", fontsize = 14)

spaak_data = pd.DataFrame({"ND":ND, "FD":FD, "EF":EF})
spaak_data["ND*FD"] = spaak_data.ND*spaak_data.FD
spaak_data["r_i"] = spaak_data.ND + spaak_data.FD -spaak_data["ND*FD"]
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
fig.tight_layout()
fig.savefig("Figure_NFD_EF.pdf")