"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Plots the regression of pigment richness, real data, purly random and 
model prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress


# contains information about biodiversity and pigment richness
datas_biodiv = {}
# contains information about EF and pigment richness
datas_EF = {}

# folder where all datas are saved
folder = "../../2_data/5. EF/"

# Datas of Estrada
###############################################################################
c_est = "red" # color for all estrada data
estrada = pd.read_csv(folder+"estrada2.csv", delimiter = ",",
                      engine = "python")
estrada = estrada.convert_objects(convert_numeric = True)
estrada = estrada[estrada.Salinity<16]

datas_biodiv["estrada"] = [estrada['SP Pigments detected by HPLC'],
             estrada['SM Phyto-plankton taxa'], c_est]

datas_EF["estrada, chl a"] = [estrada['SP Pigments detected by HPLC'].values,
         np.nansum(estrada.iloc[:,8:-3].values, axis = 1), c_est,'^']

###############################################################################
c_lab = "blue" # color for all estrada data
# Datas of Striebel
striebel = pd.read_csv(folder+"Striebel,lab.csv",delimiter = ",")
datas_biodiv["striebel, exp"] = [striebel["Pigment richness"].values, 
             np.exp(striebel["ln taxon richness"].values), c_lab]
datas_EF["striebel, exp"] = [striebel["Pigment richness"].values, 
             np.exp(striebel["ln wet mass"].values), c_lab]

# datas of striebel field
c_fie = "cyan"
striebel_field_pigs = pd.read_csv(folder+"Striebel,field,pigments.csv",
                                  delimiter= ",")
striebel_field_spec = pd.read_csv(folder+"Striebel,field,species.csv",
                                  delimiter= ",")                    
striebel_field_spec = striebel_field_spec.convert_objects(convert_numeric = 1)
r_pig = np.nansum(striebel_field_pigs.iloc[:,1:-2]>0,axis = 1)
datas_biodiv["striebel, field"] = [r_pig                
                ,np.nansum(striebel_field_spec.iloc[:,1:-1]>0,axis = 1), c_fie]
datas_EF["striebel, field"] = [r_pig, 
           1e-9*np.nansum(striebel_field_spec.iloc[:,1:-1],axis = 1), c_fie]
datas_EF["striebel, field, pigments"] = [r_pig, striebel_field_pigs["Gesamt"].values, 
         c_fie, '^']

# datas of Spaak
###############################################################################
c_sta = "darkgreen" # color for all estrada data
c_coe = "lime"
try:
    spaak.r_coex
except (AttributeError, NameError):
    spaak = pd.read_csv("data/data_random_continuous_all_with_EF.csv")
    spaak.r_coex = np.sum(spaak.iloc[:,9:19]*np.arange(1,11), axis = 1)
datas_biodiv["spaak, coex"] = [np.arange(1,21), np.array([np.mean(
            spaak.r_coex[spaak.r_pig == i]) for i in range(1,21)]), c_coe]
from biodiversity_at_start import pig_rich_av

datas_biodiv["spaak, start"] = [pig_rich_av, np.arange(1,11), c_sta]

"""
spaak_EF = pd.read_csv("EF_and_coexistence.csv")

for t in [2,3,5,10,100]:
    EF = spaak_EF["mean at day {}".format(t)]
    EF_pigs = np.array([np.mean(EF[spaak_EF.r_pig == i]) for i in range(1,21)])
    EF_pigs = EF_pigs/1e9
    datas_EF["spa, day {}".format(t)] = [np.arange(1,21), EF_pigs]"""

datas_EF["spaak, coex"] = [np.arange(1,21), 1e-9*np.array([np.mean(spaak["biovolume,50"][spaak.r_pig == i])
                     for i in range(1,21)]), c_coe]

def plot_results(dictionary, ylabel, ax_org, twinx = False, legend = True):
    ax_org.set_ylabel(ylabel)
    if twinx:
        ax_cop = ax_org.twinx()
        ax_cop.set_ylabel("Total pigment concentration")
    handles = []
    
    for i,key in enumerate(sorted(dictionary.keys())):
        ax = ax_org
        x,y,col = dictionary[key][:3]
        handles.append(mpatches.Patch(color = col, label = key))
        try:
            marker = dictionary[key][3]
            ax = ax_cop
        except IndexError:
            marker = 'o'
        
        x,y = x[np.isfinite(x*y)], y[np.isfinite(x*y)]
        
        ax.plot(x,y,linewidth = 0,marker = marker, label = key, color = col)
        ax.semilogy()
        y = np.log(y)
            
        
        slope, intercept,r,p,stderr = linregress(x,y)
        ran = np.array([min(x), max(x)])
        y_linregres = intercept + ran*slope

        y_linregres = np.exp(y_linregres)
        ax.plot(ran, y_linregres, color = col)
    if legend:
        ax_org.legend(handles = handles,loc = "best", numpoints = 1)
    ax.set_xlabel("Pigment richness")
        
fig, ax = plt.subplots(1,2,figsize = (9,9))
plot_results(datas_biodiv, "Species richness",ax[0])
print("new")
plot_results(datas_EF,r"Biovolume $[fl ml^{-1}]$",ax[1], True, False)
fig.savefig("Figure, biodiv-EF.pdf")