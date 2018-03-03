# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_name(datas,ax = None, title = None, add_1 = True):
    
    # percentiles to be computed
    perc = np.array([5,25,75,95,50])
    percentiles = np.empty((2,21,len(perc)))
    for i in range(21): # compute the percentiles
        index = np.logical_and(datas.case=="Const1", datas.r_pig == i)
        index = np.logical_and(index, np.logical_not(np.isnan(datas.s_div)))
        try:
            percentiles[0,i] = np.percentile(datas[index].s_div,perc)
        except IndexError:
            percentiles[0,i] = np.nan
        index = np.logical_and(datas.case=="Fluctuating", datas.r_pig == i)
        index = np.logical_and(index, np.logical_not(np.isnan(datas.s_div)))
        try:
            percentiles[1,i] = np.percentile(datas[index].s_div,perc)
        except IndexError:
            percentiles[1,i] = np.nan
    if add_1:
        percentiles[:,1] = 1
    
    # plot the percentiles
    color = ['orange', 'yellow', 'purple', 'blue','black']
    for i in range(len(perc)):
        star, = ax.plot(np.arange(21),percentiles[0,:,i],'*',color = color[i])
        triangle, = ax.plot(np.arange(21),percentiles[1,:,i],'^',
                            color = color[i])

    ax.legend([star,triangle],["Fluctuation", "Constant"],loc = "upper left")
    ax.set_title("{} pigments per species".format(title),fontsize = 16)
    ax.set_xlim([0.8,20.2])
    ax.set_ylim([0.8,4])

try:
    datas
except NameError:
    datas = pd.read_csv("data/data_random_continuous_all.csv")
datas_nat = datas[datas.pigments == "real"]
fig,ax = plt.subplots(3,1, sharex = True, sharey = True, figsize = (9,9))

plot_name(datas_nat[datas_nat.r_pig_spec==1],ax[1], 1)
plot_name(datas_nat[datas_nat.r_pig_spec==4],ax[2], 5, False)
plot_name(datas_nat,ax[0], "1-10")

ax[0].set_ylabel("Species richness", fontsize = 16)
ax[1].set_ylabel("Species richness", fontsize = 16)
ax[2].set_ylabel("Species richness", fontsize = 16)
ax[-1].set_xlabel("Regional pigment richness", fontsize = 16)
plt.suptitle("Species richness vs. Trait richness", fontsize = 16)
ax[1].set_xticks([1,5,10,15,20])
ax[1].set_xticks([1,5,10,15,20])


fig.savefig("Figure, trait species diversity.pdf")
