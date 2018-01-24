"""
@author: Jurg W. Spaak

plots the figure 1B
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analytical_communities as com
import analytical_r_i_continuous as ana
import seaborn as sns

n = 10000 # numer of species to compute
species = com.gen_species(n)

# compute the invasion growth rate for constant invoming light
invasion_const = np.amin(ana.constant_I_r_i(species, 125),axis = 0)

# incoming light is sinus shaped
period = 10
size = 40
I = lambda t: size*np.sin(t/period*2*np.pi)+125 #sinus
I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period
# invasion growth rate with fluctuating incoming light
invasion_fluct = np.amin(ana.continuous_r_i(species, I,period)[1], axis = 0)\
                         /period #normalize by period


plt.figure()
data = pd.DataFrame()
data["Invasion Growth Rate"] = np.append(invasion_const, invasion_fluct)
data["I_in"] = n*["Const"]+n*["Fluctuation"]
sns.violinplot(x = "I_in", y = "Invasion Growth Rate", data = data,cut = 0,
               inner = "quartile")
plt.xlabel("Incoming light condition")
plt.title("Invasion growth rates of the Huisman model")

plt.savefig("Figure, figure1a, violin plots of invasion growth rate.pdf")

maximum = round(np.amax(invasion_fluct),4)
percents = 100*np.sum(invasion_fluct>0)/n
print(("{} is the maximal invasion growthrate. {} percents of all communities"+
       "have positive invasion growth rate.").format(maximum,percents))
