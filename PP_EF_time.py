"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the main text,
for more information check the description in the main text

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

# Datas of Estrada
###############################################################################
c_est = "red" # color for all estrada data
estrada = pd.read_csv("EF,estrada.csv", delimiter = ",",
                      engine = "python")
estrada = estrada[estrada.Salinity<16]

datas_biodiv["Estrada"] = [estrada['SP Pigments detected by HPLC'],
             estrada['SM Phyto-plankton taxa'], c_est]

###############################################################################
c_lab = "blue"
# Datas of Striebel
striebel = pd.read_csv("EF,striebel,lab.csv",delimiter = ",")
datas_biodiv["Striebel, exp"] = [striebel["Pigment richness"].values, 
             np.exp(striebel["ln taxon richness"].values), c_lab]
datas_EF["Striebel, exp"] = [striebel["Pigment richness"].values, 
             np.exp(striebel["ln wet mass"].values), c_lab]

# datas of striebel field
c_fie = "cyan"
striebel_field_pigs = pd.read_csv("EF,striebel,field,pigments.csv",
                                  delimiter= ",")
striebel_field_spec = pd.read_csv("EF,striebel,field,species.csv",
                                  delimiter= ",")                    
r_pig = np.nansum(striebel_field_pigs.iloc[:,1:-2]>0,axis = 1)
datas_biodiv["Striebel, field"] = [r_pig                
                ,np.nansum(striebel_field_spec.iloc[:,1:-1]>0,axis = 1), c_fie]
datas_EF["Striebel, field"] = [r_pig, 
           1e-9*np.nansum(striebel_field_spec.iloc[:,1:-1],axis = 1), c_fie]
         
###############################################################################
c_fietz = "orange"
# Datas of Striebel
fietz = pd.read_csv("EF,fietz.csv",delimiter = ",", engine = "python")
r_pig_fietz = np.sum(fietz.iloc[:,3:28]>0, axis = 1).values
datas_biodiv["Fietz"] = [r_pig_fietz, np.nansum(fietz.iloc[:,28:].values>0
                 , axis = 1),c_fietz]
datas_EF["Fietz"] = [r_pig_fietz, np.nansum(fietz.iloc[:,28:].values, axis = 1)
             ,c_fietz]

###############################################################################
# datas of Spaak
c_sta = "lime"
c_coe = "green"
t = 240

def wheighted_median(val, weights, percentile = 50):
    val = np.asarray(val)
    weights = np.asarray(weights)
    
    ind = np.argsort(val)
    v = val[ind]
    w = weights[ind]

    pecent_50 = np.argmin(np.cumsum(w)/np.sum(w)<=percentile/100)
    return v[pecent_50]

def medians(x_val, y_val, wheight):
    # compute averages
    x_val, y_val = spaak_data[x_val], spaak_data[y_val]
    wheight = spaak_data[wheight]
    x_range = np.arange(min(x_val), max(x_val)+1)
    return x_range, np.array([wheighted_median(y_val[x_val==x], 
                        wheight[x_val ==x]) for x in x_range])

spaak_data = pd.read_csv("data/data_EF_time_all.csv")

datas_biodiv["Spaak, t="+str(t//24)] = [*medians("r_pig, start","r_spec, t="
                                   +str(t), "n_com"),c_sta]
datas_biodiv["Spaak, equi"] = [*medians("r_pig, start","r_spec, equi",
                                        "n_com"),c_coe]

for col in spaak_data.columns:
    if col[:2] == "EF":
        spaak_data[col] *=1e-8
                        
pig_range, EF_equi = medians("r_pig, start", "EF, equi", "n_com")
pig_range, EF_t = medians("r_pig, start", "EF, t="+str(t), "n_com")
datas_EF["Spaak, equi"] = [pig_range, EF_equi, c_coe]
datas_EF["Spaak, t="+str(t//24)] = [pig_range, EF_t, c_sta]

###############################################################################
# plot boxes
fs_label = 20

def plot_results(dictionary, ylabel, ax, keys):
    ax.set_ylabel(ylabel, fontsize = fs_label)
    leg = []
    for i,key in enumerate(keys):
        x,y,col = dictionary[key][:3]
        #handles.append(mpatches.Patch(color = col, label = key))
        
        x,y = x[np.isfinite(x*np.log(y))], y[np.isfinite(x*np.log(y))]
        
        leg += ax.plot(x,y,'.', label = key, color = col)
        ax.semilogy()
        y = np.log(y)
            
        
        slope, intercept,r,p,stderr = linregress(x,y)
        ran = np.array([min(x), max(x)])
        y_linregres = intercept + ran*slope

        y_linregres = np.exp(y_linregres)
        ax.plot(ran, y_linregres, color = col)
    return leg

plt.style.use('dark_background')
# plot biodiversity
fig = plt.figure(figsize = (7,7))
plt.gca().tick_params(axis = "both", which = "both", length = 6, width=2)
plt.xlim([0,24])

plt.legend(loc = "lower right", fontsize = 14)
plt.ylabel("Species richness", fontsize = fs_label)
plt.xlabel("Pigment richness", fontsize = fs_label)
plt.xticks([1,10,20], [1,10,20], fontsize = 14)

keys = ["Spaak, t="+str(t//24),"Spaak, equi",
              "Estrada", "Fietz", "Striebel, exp", "Striebel, field"]
for i in [2,1,0]:
    leg = plot_results(datas_biodiv, "Species richness", plt.gca(),
             keys[i:])
    plt.legend(leg, keys[i:], fontsize = 14)
    plt.yticks([1,10],["1","10"], fontsize = 14)
    plt.ylim([0.7,None])
    fig.savefig("PP_slides/PP, biodiv_{}.png".format(2-i))

###############################################################################
# plot EF
# plot biodiversity
fig = plt.figure(figsize = (7,7))

plt.xlim([0,24])
plt.gca().tick_params(axis = "both", which = "both", length = 6, width=2)

plt.legend(loc = "lower right", fontsize = 14)
plt.xlabel("Pigment richness", fontsize = fs_label)
plt.xticks([1,10,20], [1,10,20], fontsize = 14)

keys = ["Fietz", "Striebel, exp", "Striebel, field",
        "Spaak, t="+str(t//24),"Spaak, equi"]
for j,i in enumerate([3,5]):
    leg = plot_results(datas_EF, "Biomass [mg/ml]", plt.gca(),
             keys[:i])
    plt.legend(leg, keys[:i], fontsize = 14, loc = "upper left")
    plt.yticks([1,10],["1","10"], fontsize = 14)
    #plt.ylim([0.7,None])
    fig.savefig("PP_slides/PP, EF_{}.png".format(j+1))