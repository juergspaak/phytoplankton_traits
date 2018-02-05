"""
@author: Jurg W. Spaak

plots the figure 1 with adding photoinhibition
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import communities_analytical as com
import communities_numerical as num
import r_i_analytical_continuous as ana
import seaborn as sns

from photoinhibition import resident_density,equilibrium
from scipy.integrate import simps


# First compute the invasion growt rates for the saturating model
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

data_sat = pd.DataFrame()
data_sat["Invasion Growth Rate"] = np.append(invasion_const, invasion_fluct)
data_sat["I_in"] = n*["Const"]+n*["Fluctuation"]
data_sat["model"] = "saturating"

# Compute the invasion growth rates for photoinhibition model

def continuous_r_i(species, I,period, acc = 1001):
    # computes the boundary growth rates for invading species
    W_r_t, W_r_star,dt = resident_density(species, I, period,acc)
    i,r = [[0,1],[1,0]]
    k,l = species[[0,-1]]
    simple_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_star[:,r])-1,
                       dx = dt,axis = 0)*l[i]
    exact_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_t[:,r])-1,
                       dx = dt,axis = 0)*l[i]
    return simple_r_i, exact_r_i
    
def constant_I_r_i(species, I_in):
    # compute the boundary growth rate for constant incoming light intensity
    equi = equilibrium(species, I_in)
    k,l = species[[0,-1]]
    i,r = [0,1],[1,0]
    return l[i]*((k*equi)[i]/(k*equi)[r]-1)
    
    
species,carbon, I_r = num.gen_species(num.photoinhibition_par,50000)
n = species.shape[-1]
# incoming light is sinus shaped
period = 10
size = 450
I = lambda t: size*np.sin(t/period*2*np.pi)+550 #sinus
I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period
# invasion growth rate with fluctuating incoming light
invasion_fluct = np.amin(continuous_r_i(species, I,period)[1], axis = 0)\
                         /period #normalize by period
                         
invasion_const = np.amin(constant_I_r_i(species, 550),axis = 0)
data_pho = pd.DataFrame()
data_pho["Invasion Growth Rate"] = np.append(invasion_const, invasion_fluct)
data_pho["I_in"] = n*["Const"]+n*["Fluctuation"]
data_pho["model"] = "photoinhibition"

data = pd.concat([data_sat, data_pho])

# plot the figure

plt.figure()
sns.violinplot(x = "model", y = "Invasion Growth Rate",hue = "I_in", 
               data = data,cut = 0, inner = "quartile",split = True)
plt.xlabel("Incoming light condition")
plt.title("Invasion growth rates of the Huisman model")

plt.savefig("Figure, figure1, with photoinhibition.pdf")

for model in ["photoinhibition", "saturating"]:
    inv_gr = data[np.logical_and(data.model == model, data.I_in=="Fluctuation")]["Invasion Growth Rate"].values
    print(np.amax(inv_gr), " is the maximal invasion growthrate in the model "
          + model)
    print(np.sum(inv_gr>0)/len(inv_gr), " is the percentages of positive invasion growth rates "
          + model)