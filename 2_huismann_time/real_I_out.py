"""
@author: Jurg W. Spaak
Computes the boundary growth for both species with analytical solution
and continuous change in incoming light intensity
Does not assume, that I_out = 0

Compares result obtained with and withour assumption"""

import numpy as np
import analytical_communities as com
import analytical_r_i_continuous as ana
from numerical_r_i import own_ode
from scipy.integrate import simps

def real_I_out_r_i(species, I, period, real = True, k_back = 0, ret_I = False):
    # body of function
    # computes the boundary growth rates for invading species
    k,H,p,l = species
    def dwdt(W,t):
        k,H,p,l = species
        W_star = p/(k*l)*np.log((H+I(t))/(H+real*I(t)*np.exp(-k*W+k_back)))
        return ((k*W)/(k_back+k*W)*W_star-W)*l
    
    # find approximate starting densities of species
    W_start, W_r_star = ana.resident_density(species,I,period,1001)[:2]
    
    # run n_per periods numiercally to find real equilibrium densities
    n_per = 10
    time = np.array([0, n_per*period])
    
    # Compute real starting density of resident species
    sols = own_ode(dwdt,W_start[0], time, steps = n_per*100)
    
    # compute resident densities over entire period with increase accuracy
    time_inv = np.array([0,period])
    iters = 501 # number of time steps
    res_dist = own_ode(dwdt, sols[-1], time_inv, steps = iters)
    
    I_in = I(np.linspace(*time_inv, iters)).reshape(-1,1,1)
    r_ind = [1,0] # index of resident species
    I_out = I_in*np.exp(-k[r_ind]*res_dist[:,r_ind])
    inv_star = p/k*np.log((I_in+H)/(I_out+H))/l
    
    
    growth = (k*inv_star/(k[r_ind]*res_dist[:,r_ind]+k_back)-1)*l
    invasion_growth = simps(growth, dx = period/iters,axis = 0)
    if ret_I:
        return invasion_growth, I_out
    else:
        return invasion_growth


if __name__  == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    n = 1000
    species = com.gen_species(n)
    k,H,p,l = species
    
    # I_in over time
    # incoming light is sinus shaped
    period = 10
    size = 40
    I = lambda t: size*np.sin(t/period*2*np.pi)+125 #sinus
    I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period
    
    # compute the invasions growth rates
    # actual invasion growth rate, no assumption
    invasion_real, I_out = real_I_out_r_i(species, I, period, ret_I = True)
    # same function, assuming that I_out = 0
    invasion_real_check = real_I_out_r_i(species, I, period, False)
    # different function, assuming that I_out = 0
    invasion_approx = ana.continuous_r_i(species, I,period)[1]
    
    # plot I_out
    plt.figure()
    plt.semilogy(np.linspace(0,100,I_out.size),sorted(I_out.ravel()))
    plt.xlabel("percentiles")
    plt.ylabel("I_out")
    plt.axis([0,100,10**-10, 10**1])

    # plot relative differences between invasion growth rates
    plt.figure()
    # relative difference computation
    sorted_diff = lambda x,y: sorted(np.abs((x.ravel()-y.ravel())/x.ravel()))
    percents = np.linspace(0,100,2*n)
    plt.plot(percents,sorted_diff(invasion_approx, invasion_real),
             label = "approx vs. real")
    plt.plot(percents,sorted_diff(invasion_approx, invasion_real_check),
             label = "approx vs. check")
    plt.plot(percents,sorted_diff(invasion_real, invasion_real_check),
             label = "check vs. real")
    plt.semilogy()
    plt.legend()
    plt.ylabel("relative difference")
    plt.xlabel("percentiles")
    
    # compute invvasion of inferior species
    invasion_real = np.amin(invasion_real,axis = 0)/period
    invasion_approx = np.amin(invasion_approx, axis = 0)/period
    invasion_real_check = np.amin(invasion_real_check, axis = 0)/period
    
    # plot in similar style to figure 1a
    plt.figure()
    data = pd.DataFrame()
    data["Invasion Growth Rate"] = np.append(np.append(invasion_approx, 
                                    invasion_real), invasion_real_check)
    data["I_in"] = n*["Approx"]+n*["Real"]+n*["Check"]
    sns.violinplot(x = "I_in", y = "Invasion Growth Rate", data = data,cut = 0,
                   inner = "quartile")
    plt.xlabel("Model assumptions")
    plt.title("Invasion growth rates of the Huisman model")
    plt.show()
    
    # print quantitative results
    maximum = round(np.amax(invasion_approx),4)
    percents = 100*np.sum(invasion_approx>0.0001)/n
    print(("{} is the maximal invasion growthrate. {} percents of all communities"+
           " have positive invasion growth rate with approximation.").format(maximum,percents))
    maximum = round(np.amax(invasion_real),4)
    percents = 100*np.sum(invasion_real>0.0001)/n
    print(("{} is the maximal invasion growthrate. {} percents of all communities"+
           " have positive invasion growth rate with real model.").format(maximum,percents))
    maximum = round(np.amax(invasion_real_check),4)
    percents = 100*np.sum(invasion_real_check>0.0001)/n
    print(("{} is the maximal invasion growthrate. {} percents of all communities"+
           " have positive invasion growth rate with check function.").format(maximum,percents))
