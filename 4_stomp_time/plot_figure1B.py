# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# datas to use
dat_case = "org"
datas = pd.read_csv("data/data_random_{}_all.csv".format(dat_case))
# compute average for each community
num_data = np.array(datas[[str(i+1) for i in range(10)]])
ave_data = (np.arange(1,11)*num_data).sum(axis = -1)
datas["s_div"] = ave_data

# percentiles to be computed
perc = np.array([5,25,50,75,95])
percentiles = np.empty((2,20,len(perc)))
a = []
for i in range(20): # compute the percentiles
    index = np.logical_and(datas.case=="Const1", datas.r_pig == i+1)
    try:
        percentiles[0,i] = np.percentile(datas[index].s_div,perc)
    except IndexError:
        percentiles[0,i] = np.nan

    index = np.logical_and(datas.case=="Fluctuating", datas.r_pig == i+1)
    try:
        percentiles[1,i] = np.percentile(datas[index].s_div,perc)
    except IndexError:
        percentiles[1,i] = np.nan

# plot the percentiles
color = ['yellow', 'orange', 'red', 'purple', 'blue']
fig, ax = plt.subplots(figsize = (9,7))
for i in range(len(perc)):
    star, = ax.plot(np.arange(20),percentiles[0,:,i],'*',color = color[i])
    triangle, = ax.plot(np.arange(20),percentiles[1,:,i],'^',color = color[i])
ax.set_ylabel("species diversity")
ax.set_xlabel("trait diversity")
ax.legend([star,triangle],["Fluctuation", "Constant"],loc = "upper left")
fig.savefig("Figure, Figure1.pdf")