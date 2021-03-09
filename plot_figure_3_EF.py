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

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16) 

# contains information about biodiversity and pigment richness
datas_biodiv = {}
# contains information about EF and pigment richness
datas_EF = {}

# Datas of Estrada
###############################################################################
c_est = "red" # color for all estrada data
estrada = pd.read_csv("EF,estrada.csv", delimiter = ",",
                      engine = "python")
estrada = estrada[estrada.Salinity<23]

pigments_est = estrada.keys()[8:-3] # all pigments
# only photosynthetic pigments
pigments_est = ["Chl c", "Per", "Fuc", "Chl b", "Chl c", "beta-Carotene"]
est_pig_richness = np.sum(estrada[pigments_est]>0, axis = 1)

datas_biodiv["Estrada"] = [estrada['SP Pigments detected by HPLC'].values,
             estrada['SM Phyto-plankton taxa'].values, c_est]
#datas_biodiv["Estrada2"] = [est_pig_richness.values,
#             estrada['SM Phyto-plankton taxa'].values, "darkred"]

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

#############
# field data striebel
striebel_field_pigs["Chl c"] = np.sum(striebel_field_pigs[["Chlorophyll c1",
                    "Chlorophyll c2", "Chlorophyll c3"]], axis = 1)
pigments_striebel = ["Chl c", "Chlorophyll b", "Total Chlorophyll a", 
                     "Fucoxanthin", "Peridinin", "b-carotene", ]
r_pig = np.nansum(striebel_field_pigs[pigments_striebel]>0,axis = 1)
#datas_biodiv["Striebel, field2"] = [r_pig                
#                ,np.nansum(striebel_field_spec.iloc[:,1:-1]>0,axis = 1), "grey"]
#datas_EF["Striebel, field2"] = [r_pig, 
#           1e-9*np.nansum(striebel_field_spec.iloc[:,1:-1],axis = 1), "grey"]
         
###############################################################################
c_fietz = "orange"
# Datas of Striebel
fietz = pd.read_csv("EF,fietz.csv",delimiter = ",", engine = "python")

r_pig_fietz = np.sum(fietz.iloc[:,3:28]>0, axis = 1).values
datas_biodiv["Fietz"] = [r_pig_fietz, np.nansum(fietz.iloc[:,28:].values>0
                 , axis = 1),c_fietz]
datas_EF["Fietz"] = [r_pig_fietz, np.nansum(fietz.iloc[:,28:].values, axis = 1)
             ,c_fietz]
pigments_fietz = ["Peridinin", "Fuco", "b-Car", "Chl a", "Chl b", "Chl c", 
                  "Ph a", "Ph b"]
"""
r_pig_fietz = np.sum(fietz[pigments_fietz]>0, axis = 1).values
datas_EF["Fietz2"] = [r_pig_fietz
        , np.nansum(fietz.iloc[:,28:].values, axis = 1)
             ,"darkorange"]

datas_biodiv["Fietz2"] = [r_pig_fietz, np.nansum(fietz.iloc[:,28:].values>0
                 , axis = 1),"darkorange"]
"""
###############################################################################
# datas of Spaak
c_sta = "lime"
c_coe = "green"
t = 240

try:
    spaak_data
except NameError:
    from load_data import data_org, max_spec

spaak_data = data_org[data_org.r_spec_start >= max_spec]

def metric(X, fun = np.nanmean):
    return np.array([fun(x) for x in X])

EF, EF_init = [],[]
r_spec, r_spec_init = [], []
for i,n in enumerate(sorted(set(spaak_data["r_pig_start"]))):
    ind = spaak_data["r_pig_start"] == n
    EF.append(spaak_data["EF_equi"][ind])
    EF_init.append(spaak_data["EF_t={}".format(t)][ind])
    r_spec_init.append(spaak_data["r_spec_t={}".format(t)][ind].astype(int).values)
    r_spec.append(spaak_data["r_spec_equi"][ind].astype(int).values)

pr = np.arange(min(spaak_data.r_pig_start), max(spaak_data.r_pig_start)+1)
datas_biodiv["Spaak, t="+str(t//24)] = [pr,metric(r_spec_init), c_sta]
datas_biodiv["Spaak, equi"] = [pr, metric(r_spec),c_coe]

datas_EF["Spaak, equi"] = [pr, metric(EF, np.nanmedian), c_coe]
datas_EF["Spaak, t="+str(t//24)] = [pr, metric(EF_init, np.nanmedian), c_sta]

###############################################################################
# plot boxes
def boxs(x_val, y_val, x_range,ax,color):
    # compute averages
    x_val, y_val = spaak_data[x_val], spaak_data[y_val]
    def_col = dict(color= color)
    ax.boxplot([y_val[x_val==x] for x in x_range], boxprops = def_col,
               whiskerprops = def_col, capprops = def_col,
               medianprops = def_col, showfliers = False)

def plot_results(dictionary, ylabel, ax, legend = True):
    ax.set_ylabel(ylabel)
    handles = []
    
    for i,key in enumerate(sorted(dictionary.keys())):
        x,y,col = dictionary[key][:3]
        handles.append(mpatches.Patch(color = col, label = key))
        
        x,y = x[np.isfinite(x*np.log(y))], y[np.isfinite(x*np.log(y))]
        
        ax.plot(x,y,'.', label = key, color = col)
        ax.semilogy()
        y = np.log(y)
            
        
        slope, intercept,r,p,stderr = linregress(x,y)
        #print(key, (20-len(key))*" ", 
        #      np.round([slope, intercept, r,p, stderr],3))
        print(key, (20-len(key))*" ", 
              np.round([slope, p, stderr],3))
        if p>0.05:
            linestyle = "--"
        else:
            linestyle = "-"
        ran = np.array([min(x), max(x)])
        y_linregres = intercept + ran*slope

        y_linregres = np.exp(y_linregres)
        ax.plot(ran, y_linregres, color = col, linestyle = linestyle)
    if legend:
        ax.legend(loc = "best", numpoints = 1)
        
fig, ax = plt.subplots(2,2,figsize = (12,9), sharex = "col", sharey = "row")
ax[0,0].set_title("A")
ax[0,1].set_title("C")
ax[1,0].set_title("B")
ax[1,1].set_title("D")

pig_range = range(1,24)

plot_results(datas_biodiv, "Species richness",ax[0,0])
print("\n\n", "EF", "\n\n")
plot_results(datas_EF,r"Biovolume $[fl\,ml^{-1}]$",ax[1,0], False)
ax[0,0].set_xlim(0.5,23.5)
ax[0,0].set_xticks(range(2,24,2),range(2,24,2))
ax[1,0].set_xlabel("initial\npigment richness")

###############################################################################
# plot size dependency
print("\n\n", "size", "\n\n")
x = "size_cv"

x_dat = spaak_data[x].values
ranges,dr = np.linspace(*np.nanpercentile(x_dat, [1,99]), 27,
                        retstep = True)


EF_equi = []
EF_mid = []
rich_equi = []
rich_mid = []
for i in range(len(ranges)-1):
    ind = (x_dat>=ranges[i]) & (x_dat<ranges[i+1])
    
    rich_equi.append(spaak_data["r_spec_equi"][ind].values)
    rich_mid.append(spaak_data["r_spec_t=240"][ind].values)
    
    EF_equi.append(spaak_data["EF_equi"][ind].values)
    EF_mid.append(spaak_data["EF_t=240"][ind].values)
    
EF_equi = np.array([np.nanmedian(i) for i in EF_equi])
EF_mid = np.array([np.nanmedian(i) for i in EF_mid])
rich_mid = np.array([np.nanmean(i) for i in rich_mid])
rich_equi = np.array([np.nanmean(i) for i in rich_equi])

datas_rich_EF = {}
datas_rich_EF["Spaak, equi"] = [ranges[1:], rich_equi, c_coe]
#datas_rich_EF["Spaak, t=10"] = [ranges[1:], rich_mid, c_sta]
plot_results(datas_rich_EF, "",ax[0,1], legend = False)
ax[0,1].plot(ranges[1:], rich_mid, ".", color = c_sta)
print("\n\n", "EF_size", "\n\n")
datas_size_EF = {}
datas_size_EF["Spaak, equi"] = [ranges[1:], EF_equi, c_coe]
datas_size_EF["Spaak, t=10"] = [ranges[1:], EF_mid, c_sta]
plot_results(datas_size_EF, "",ax[1,1], legend = False)

ax[1,1].axhline(1, color = "k", zorder = 0)
ax[1,0].axhline(1, color = "k", zorder = 0)
ax[0,1].axhline(1, color = "k", zorder = 0)
ax[0,0].axhline(1, color = "k", zorder = 0)

ax[1,1].set_xlabel("Initial\nCV(size)")
fig.tight_layout()

fig.savefig("Figure_3_biodiv_EF.pdf")


