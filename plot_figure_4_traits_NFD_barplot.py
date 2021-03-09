# -*- coding: utf-8 -*-
"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot the main Figure of the paper
See description in manuscript for more information
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# load the dataset
try:
    data_org
except NameError:
    from load_data import data_org, max_spec
    
data_ND = data_org[data_org.r_spec_equi != 1]

data_ND = data_ND[np.isfinite(data_ND.size_cv)]
data_ND = data_ND[data_ND.r_spec_start >= max_spec]
datas = {key: data_ND.copy() for key in ["Initial", "Equilibrium"]}
datas["Equilibrium"].size_cv = datas["Equilibrium"].size_sur_cv
datas["Equilibrium"].r_pig_start = datas["Equilibrium"].r_pig_equi


ran = {"Equilibrium": 0.6, "Initial": 3.5}
xticks = {"Equilibrium": [0,0.25,0.5],
          "Initial": [0,1,2,3]}
###############################################################################

for name in datas.keys():
    data_ND = datas[name]
    r_pig = np.arange(min(data_ND.r_pig_start), max(data_ND.r_pig_start)+1)
    pig_range = np.arange(min(data_ND.r_pig_start),
                          max(data_ND.r_pig_start) +1)
    
    
    ND_cols = ["ND_{}".format(i) for i in range(5)]
    ND = data_ND[ND_cols].values
    
    FD_cols = ["FD_{}".format(i) for i in range(5)]
    FD = data_ND[FD_cols].values
    
    ND_box = [ND[data_ND.r_pig_start == i].flatten() for i in r_pig]
    ND_box = [N[np.isfinite(N)] for N in ND_box]
    FD_box = [-FD[data_ND.r_pig_start == i].flatten() for i in r_pig]
    FD_box = [F[np.isfinite(F)] for F in FD_box]
    FD_box = [np.abs(F) for F in FD_box]
    
    
    x = "r_pig_start"
    y = ND_cols
    
    
    def nan_linreg(x,y, ax_c, x_lim):
        x = data_ND[x].values
        y = data_ND[y].values
        if y.ndim != 1:
            x = x[:,np.newaxis]*np.ones(y.shape)
        ind = np.isfinite(x*y)
        x = x[ind]
        y = y[ind]
        slope, intercept, r_value, p, stderr = linregress(x,y)
        return (slope, intercept, r_value**2, p, stderr)
    
    ###########################################################################
    # plotting results
    fig, ax = plt.subplots(2,2,figsize = (7,7), sharey = "row", sharex = "col")
    ax[0,0].boxplot(ND_box, sym = "", positions = r_pig)
    nan_linreg("r_pig_start", ND_cols, ax[0,0], r_pig)
    ax[1,0].boxplot(FD_box, sym = "", positions = r_pig)
    nan_linreg("r_pig_start", ["FD_abs_{}".format(i) for i in range(5)]
            , ax[1,0], r_pig)
    
    ax[0,0].set_ylabel("$\mathcal{N}_i$")
    ax[1,0].set_ylabel("$|\mathcal{F}_i|$")
    ax[1,0].set_xlabel("{}\npigment richness".format(name))
    
    ###########################################################################
    # plot of sive variation
    x = "size_cv"
    
    
    x_dat = data_ND[x].values
    ranges,dr = np.linspace(min(x_dat), ran[name], 16,
                            retstep = True)
    
    ND_cols = ["ND_{}".format(i) for i in range(5)]
    FD_cols = ["FD_{}".format(i) for i in range(5)]
    ND_box = []
    FD_box = []
    for i in range(len(ranges)-1):
        ind = (x_dat>ranges[i]) & (x_dat<ranges[i+1])
        ND_box.append(data_ND[ND_cols][ind].values)
        ND_box[-1] = ND_box[-1][np.isfinite(ND_box[-1])]
        FD_box.append(data_ND[FD_cols][ind].values)
        FD_box[-1] = FD_box[-1][np.isfinite(FD_box[-1])]
    FD_box = [np.abs(F) for F in FD_box]    
    ax[0,1].boxplot(ND_box, positions = ranges[1:] - dr/2, sym = "",
      widths = 0.8*dr)
    nan_linreg("size_cv", ND_cols
            , ax[0,1], ranges[1:]-dr/2)
    
    ax[1,1].boxplot(FD_box, positions = ranges[1:] - dr/2, sym = "",
      widths = 0.8*dr)
    print([len(i) for i in FD_box])
    nan_linreg("size_cv", ["FD_abs_{}".format(i) for i in range(5)]
            , ax[1,1], ranges[1:]-dr/2)
    ax[1,1].set_xlabel("{}\nCV(size)".format(name))
    
    ax[0,0].set_title("A")
    ax[0,1].set_title("C")
    ax[1,0].set_title("B")
    ax[1,1].set_title("D")
    
    
    ax[1,0].set_xticks([r_pig[0],5,10,r_pig[-1]])
    ax[1,0].set_xticklabels([r_pig[0],5,10,r_pig[-1]])
    
    ax[0,0].set_yticks([0,0.1,0.2])
    ax[1,0].set_yticks([0, 0.1])
    
    ticks = xticks[name]
    ax[1,1].set_xticks(ticks+dr/2)
    ax[1,1].set_xticklabels(ticks)
    ax[1,1].set_xlim(ticks[0]-dr/2, ranges[-1] +dr/2)
    
    fig.savefig("Figure_4_traits_NFD_barplot_{}.pdf".format(name))