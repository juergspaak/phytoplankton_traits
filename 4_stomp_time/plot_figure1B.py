# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_name(dat_case = "org",ax = None):

    datas = pd.read_csv("data/data_random_{}_all.csv".format(dat_case))
    datas = datas[datas["pigments"]=="rand"]
    # compute average for each community
    num_data = np.array(datas[[str(i+1) for i in range(10)]])
    ave_data = (np.arange(1,11)*num_data).sum(axis = -1)
    datas["s_div"] = ave_data
    
    # percentiles to be computed
    perc = np.array([5,25,50,75,95])
    percentiles = np.empty((2,21,len(perc)))
    for i in range(21): # compute the percentiles
        index = np.logical_and(datas.case=="Const1", datas.r_pig == i)
        try:
            percentiles[0,i] = np.percentile(datas[index].s_div,perc)
        except IndexError:
            print(i)
            percentiles[0,i] = np.nan
    
        index = np.logical_and(datas.case=="Fluctuating", datas.r_pig == i)
        try:
            percentiles[1,i] = np.percentile(datas[index].s_div,perc)
        except IndexError:
            percentiles[1,i] = np.nan
    percentiles[:,1] = 1
    
    # plot the percentiles
    color = ['yellow', 'orange', 'red', 'purple', 'blue']

    for i in range(len(perc)):
        star, = ax.plot(np.arange(21),percentiles[0,:,i],'*',color = color[i])
        triangle, = ax.plot(np.arange(21),percentiles[1,:,i],'^',color = color[i])

    ax.legend([star,triangle],["Fluctuation", "Constant"],loc = "upper left")
    ax.set_title(dat_case)
    ax.set_xlim([0.8,20.2])
    ax.set_ylim([0.8,4])

fig, ax = plt.subplots(2,2,sharex = True, sharey = True,figsize = (11,9))
fig.add_subplot(111,frameon = False)
plt.tick_params(labelcolor="none",top="off",bottom="off",
                left="off",right="off")
plt.ylabel("Species diversity", fontsize = 16)
plt.xlabel("Trait diversity", fontsize = 16)
plt.grid(False)
plt.suptitle("Species diversity vs Trait diversity", fontsize = 16)
plot_name("org",ax[0,0])
plot_name("continuous", ax[1,0])
plot_name("comp_light", ax[1,1])
plot_name("step", ax[0,1])

fig.savefig("Figure, figure 1B.pdf")
