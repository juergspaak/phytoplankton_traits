import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.integrate import simps, odeint
from timeit import default_timer as timer

import richness_computation as rc

from I_in_functions import I_in_def
from load_pigments import dlam

time = np.arange(500)

def find_EF(r_spec, r_pig, r_pig_spec, n_com):
    fac = 2
    [phi, l], k_spec, alpha = rc.gen_com(r_pig,r_spec,r_pig_spec, 
                    fac,n_com = n_com)
    
    start_dens = np.full(phi.shape, 1.0)
    start_dens *= 1e9/np.sum(start_dens, axis = 0)
    
    I_in = lambda t: I_in_def(40)
    # solve with odeint
    def multi_growth(N_r,t):
        N = N_r.reshape(-1,n_com)
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = np.nansum(N*k_spec, axis = 1, keepdims = True)
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
        return (N*(growth-l)).flatten()
    
    sol_ode = odeint(multi_growth, start_dens.reshape(-1), time)
    sol2 = sol_ode.reshape(len(time),r_spec,n_com)
    return sol2

   
n_com = 50
iters = 10
r_pig = np.random.randint(1,20,2*iters)
r_spec = np.random.randint(5,41,iters)
r_pig_spec = np.random.randint(1,20,2*iters)
goods = r_pig>=r_pig_spec
r_pig, r_pig_spec = r_pig[goods][:iters], r_pig_spec[goods][:iters]
sols = np.zeros((iters, len(time),n_com))
richness = np.empty((iters, len(time), n_com))

def saveit(sols, richness,i):
    sols_mean = np.nanmean(sols, axis = -1)
    sols_var = np.nanvar(sols, axis = -1)
    df = pd.DataFrame()
    df["r_pig"] = r_pig[:i+1]
    df["r_spec"] = r_spec[:i+1]
    df["r_pig_spec"] = r_pig_spec[:i+1]
    
    for j,t in enumerate(time):
        df["mean at day {}".format(t)] = sols_mean[:,j]
    for j,t in enumerate(time):
        df["var at day {}".format(t)] = sols_var[:,j]
    df.to_csv("EF_and_coexistence.csv")
    
    rich_mean = np.nanmean(richness, axis = -1)
    rich_var = np.nanvar(richness, axis = -1)
    df["r_pig"] = r_pig[:i+1]
    df["r_spec"] = r_spec[:i+1]
    df["r_pig_spec"] = r_pig_spec[:i+1]

    for j,t in enumerate(time):
        df["mean at day {}".format(t)] = rich_mean[:,j]
    for j,t in enumerate(time):
        df["var at day {}".format(t)] = rich_var[:,j]
    df.to_csv("EF_and_coexistence_richness.csv") 

start = timer()
for i in range(iters):
    
    EF = find_EF(r_spec[i], r_pig[i], r_pig_spec[i], n_com)
    sols[i] = np.nansum(EF, axis = 1)
    richness[i] = np.nansum(EF>1e7, axis = 1)
    if i%100==99:
        print(i,r_spec[i], r_pig[i], r_pig_spec[i], timer()-start)
        saveit(sols[:i+1], richness[:i+1],i)
        
sols_mean = np.nanmean(sols, axis = -1)
sols_var = np.nanvar(sols, axis = -1)
df = pd.DataFrame()
df["r_pig"] = r_pig
df["r_spec"] = r_spec
df["r_pig_spec"] = r_pig_spec

for i,t in enumerate(time):
    df["mean at day {}".format(t)] = sols_mean[:,i]
for i,t in enumerate(time):
    df["var at day {}".format(t)] = sols_var[:,i]
df.to_csv("EF_and_coexistence.csv")

rich_mean = np.nanmean(richness, axis = -1)
rich_var = np.nanvar(richness, axis = -1)
df = pd.DataFrame()
df["r_pig"] = r_pig
df["r_spec"] = r_spec
df["r_pig_spec"] = r_pig_spec

for i,t in enumerate(time):
    df["mean at day {}".format(t)] = rich_mean[:,i]
for i,t in enumerate(time):
    df["var at day {}".format(t)] = rich_var[:,i]
df.to_csv("EF_and_coexistence_richness.csv") 

EF_av = np.nanmean(sols, axis = -1)

datas = [r_pig, r_spec, r_pig_spec]
labels = ["r_pig", "r_spec", "r_pig_spec", "time [days]"]

for i in range(3):
    first = datas[i]
    f_max = max(first)
    for j in range(3):
        if j<=i:
            continue
        sec = datas[j]
        s_max = max(sec)
        EF_data = np.empty((f_max, s_max))
        with warnings.catch_warnings():
            for f in range(f_max):
                for s in range(s_max):
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ind = (first==f) & (sec==s)
                    EF_data[f,s] = np.mean(EF_av[ind,-1], axis = 0)
        plt.figure()
        plt.imshow(EF_data, cmap = "hot", interpolation = "nearest", 
                   aspect = "auto")
        plt.xlabel(labels[j])
        plt.ylabel(labels[i])
        plt.colorbar()

max_r_pig = max(r_pig)
EF_all = np.empty((len(time), max_r_pig))
for j in range(len(time)):
    EF_all[j] = [np.nanmean(EF_av[r_pig==i+1,j], axis = 0) 
            for i in range(max_r_pig)]
plt.figure(figsize = (8,7))
plt.imshow(EF_all, cmap = "hot", interpolation = "bilinear", aspect = "auto")
plt.xlabel("pigment richness")
plt.ylabel("time [days]")

plt.colorbar()

fig = plt.figure(figsize = (8,7))
plt.pcolor(EF_all[2:], cmap = "hot")
plt.xlabel("pigment richness")
plt.ylabel("time [days]")

plt.colorbar()
fig.savefig("EF_time_r_pig.pdf")

