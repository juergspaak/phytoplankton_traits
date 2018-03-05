# -*- coding: utf-8 -*-
"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Plots the regression of pigment richness, real data, purly random and 
model prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
species_id = "eawag-data"
if species_id == "rand":
    # probability distribution of number of pigments
    p_imp = np.array([5,8,15,16,16,16,16])/16 # only important pigments
    p_ave = np.array([1,4,5,11,14,15,16])/16 # of average importance
    p_all = np.array([1,2,4,7,12,15,16])/16 # all pigments
    p = p_all
    r_pig = 18 # number of pigments in total
    r_spe = 100 # number of species
    
    # determine which species has how many pigments
    spe_id = np.random.uniform(size = r_spe)
    # number of pigments for each species
    r_pig_spe = 4 + np.sum(np.array([spe_id>=pi for pi in p]),axis = 0)
    # pigment identity in each species
    pig_spe_id = np.zeros(shape = (r_spe, r_pig))
    new = np.random.rand(r_spe, r_pig).argsort(axis = 1)
    # randomly allocate pigments in species
    for i in range(r_spe):
        pig_spe_id[i,new[i,:r_pig_spe[i]]] = 1
elif species_id == "eawag-data":
    pig_spe_id = np.genfromtxt("pig_spe_id.csv",
                               delimiter = ";")[1:-1,1:-1]
    
    # pigments at 3,6,8 are for structurizing only
    pig_spe_id = pig_spe_id[np.array([0,1,2,4,5,7,8]+list(range(10,23)))]
    pig_spe_id = pig_spe_id >0.1
    r_pig, r_spe = pig_spe_id.shape
    
n_exp = 10000 #number of experiments for each richness of species
max_r_exp = 10 # maximal richness in all experiments

r_range = np.arange(1, max_r_exp+1) # range of all richness of species
r_pig_exp = np.empty((len(r_range), n_exp)) # richness of pigments in each exp
for r_exp in r_range: # r_exp is species richness in each experiment
    # which species are present
    spe_pre = np.argpartition(np.random.rand(n_exp, r_spe)
                                    ,r_exp,axis = 1)[:,:r_exp]
    # which pigments are present in the present species
    new_2 = np.array([pig_spe_id[:,i] for i in spe_pre])
    # sum over species to get pigment presence
    r_pig_exp[r_exp-1] = np.sum(np.sum(new_2,axis = 2)>0,axis = 1)

# prob. of finding i pigments in a community of j species is richnesses[i,j]
richnesses = np.array([np.sum(r_pig_exp==r+1, axis =1) for r in 
                        range(r_pig)])
richnesses = richnesses/np.sum(richnesses, axis = 0)
# preliminary plot, distribution of pigment richness
for i in range(richnesses.shape[-1]):
    plt.plot(np.arange(1,r_pig+1),richnesses[:,i], label = i)
    
# compute linear regression model
trans = lambda x:np.log(x)
log_r_range = trans(r_range)
av_spe = np.average(log_r_range) #average of species richness
var_spe = np.average((log_r_range-av_spe)**2 ) #variane in sp. richness


# compute the averages of pigment richness for each species richness
av_pig = np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis])/len(r_range)
var_pig = np.sum(richnesses*(np.arange(1,r_pig+1)[:,np.newaxis]-av_pig)**2\
                /np.sum(richnesses))
# compute the covariance of species rich. and pigment rich.
cov = richnesses*(np.arange(1,r_pig+1)[:,np.newaxis]-av_pig)*\
                (log_r_range-av_spe)/np.sum(richnesses)

beta = np.sum(cov)/var_pig
alpha = av_spe -beta*av_pig
fig, ax = plt.subplots(1,2,figsize = (11,7))

###############################################################################
#plot real data
# real data from paper
n_spe = np.array([1,1,1,2,2,2,3,3,5,5,7,7,7,10,10]) #number of species in exp.
n_pig = np.array([6,8,12,8,11,12,11,14,15,16,11,14,16,17,18]) #num. of pigments

#linear regression
slope, intercept, r, p, stderr = linregress(n_pig,trans(n_spe))
ax[0].plot(n_pig, trans(n_spe),  'go', label = "experimental")
ax[0].plot(range(6,20), intercept+slope*np.arange(6,20),'g')
ax[0].axis([ 5,19,trans(0.8), trans(20)])
ax[0].set_xlabel("Pigment richness", fontsize = 16)
ax[0].set_ylabel("Species richness (log)", fontsize = 16)
ax[0].set_title("A", fontsize = 16)
fig.savefig("Figure, Regression of pigments, real data")

###############################################################################
# plot simulated data
ax[0].plot( np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis],axis = 0),
         log_r_range,'ro', label = "theory, short term")
pig_rich_av = np.sum(richnesses*np.arange(1,r_pig+1)[:,np.newaxis],axis = 0)
slope, intercept, r, p, stderr = linregress(pig_rich_av,log_r_range)
ax[0].plot(range(6,20), intercept+slope*np.arange(6,20),'r:')
ax[0].plot(np.arange(r_pig) , alpha+beta*np.arange(r_pig),'r')

try:
    datas_EF
except NameError:
    datas_EF = pd.read_csv("data/data_random_continuous_all_with_EF.csv")
means = np.zeros(20)
for i in range(20): # compute the percentiles
    index = np.logical_and(datas_EF.case=="Const1", datas_EF.r_pig == i+1)
    index = np.logical_and(index, np.logical_not(np.isnan(datas_EF.s_div)))
    means[i] = np.median(datas_EF[index].s_div)

ax[0].plot(np.arange(r_pig), trans(means),'^',color = "black", 
                        label = "theory, long term")

ax[0].plot(np.arange(r_pig), trans(np.arange(r_pig)),'c',label = "maximum")


ax[0].legend(loc = "upper left")

print(alpha, beta)
print(intercept, slope, r)
###############################################################################
# plot panel B

def plot_name(datas,y_data = "biovolume,50",ax = None, title = None):
    
    # percentiles to be computed
    perc = np.array([5,95,50])
    percentiles = np.empty((21,len(perc)))
    for i in range(21): # compute the percentiles
        index = np.logical_and(datas.case=="Const1", datas.r_pig == i)
        index = np.logical_and(index, np.logical_not(np.isnan(datas[y_data])))
        try:
            percentiles[i] = np.percentile(datas[index][y_data],perc)
        except IndexError:
            percentiles[i] = np.nan
    
       # plot the percentiles
    color = ['orange',  'blue', 'black']
    for i in range(len(perc)):
        star, = ax.plot(np.arange(21),percentiles[:,i],'^',color = color[i])

    #ax.legend([star, triangle],["Constant", "fluctuating"],loc = "upper left")
    ax.set_title("{} pigments per species".format(title),fontsize = 16)
    ax.set_xlim([0.8,20.2])
    #ax.semilogy()

try:
    datas_EF
except NameError:
    datas_EF = pd.read_csv("data/data_random_continuous_all_with_EF.csv")
datas_nat_EF = datas_EF[datas_EF.pigments == "real"]

datas_nat_EF["absorbance"] = np.log(datas_nat_EF["lux1"]/
                datas_nat_EF["I_out,50"])/datas_nat_EF["biovolume,50"]
plot_name(datas_nat_EF,"absorbance",ax[1], "1-10")

ax[1].set_ylabel("Total absorbance (per cell)", fontsize = 16)

ax[1].set_xlabel("Pigment richness", fontsize = 16)
ax[1].set_title("B", fontsize = 16)
ax[1].set_xticks([1,5,10,15,20])

striebel_data = np.genfromtxt("../../2_data/3. Different pigments/"
                +"absorbance_pigment_richness_real_data.csv",delimiter = ",")

ax_striebel = ax[1].twinx()
ax_striebel.plot(*([[1,300]]*striebel_data).T,'go')
# add linear regession to striebel
pig_val = np.linspace(6,18,10)

ax_striebel.plot(pig_val, 300*0.005*(pig_val-1),'g')
ax_striebel.set_ylabel("Total absorbance (per mg C)", fontsize = 16)


fig.savefig("Figure, EF vs trait richness.pdf")
    