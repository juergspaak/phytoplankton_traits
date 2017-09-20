"""
@author: Jurg W. Spaak
Computes the boundary growth for both species with analytical solution
Assumes that I_out = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import analytical_communities as com
from scipy.integrate import simps
from timeit import default_timer as timer

T = 1000
size = 50
frequency = 10
def step_function(t,x=0,var=10):
    """returns a function, that is 0 on [-inf, x] and 1 on [x+var, inf]""" 
    try:
        sh = t.shape
    except:
        sh =1
    slow_start = lambda s: np.exp(-np.divide(1,s,out = np.full(sh,np.inf)
                                    , where = s>0))
    return slow_start(t-x)/(slow_start(var-t+x)+slow_start(t-x))
    
I = lambda t: size*np.sin(t/T*frequency*2*np.pi)*step_function(T*1.01-t)+100
I.dt = lambda t: (I(t+0.000001)-I(t))/0.000001
plt.plot(np.linspace(T/2, 1.2*T, 1000), I.dt(np.linspace(T/2, 1.2*T, 1000)))
species = com.gen_species(20)
bal = com.find_balance(species)

def ana_r_i(species, I):
    """ Computes the boundary growth rates of both species
    
    r_i = (k[i]/k[r]E(W*[i]/W*[r])-1)*l[i]
    where W*[i] is the equilibrium density and the average is taken over the 
    incoming light density"""
    # compute the equilibrium density of all species for all incoming lights
    W_star = com.equilibrium(species, I)
    i = [0,1]
    r = [1,0]
    k,l = species[[0,-1]]
    rel_W_star = W_star[:,i]/W_star[:,r]
    r_i = (k[i]/k[r]*np.average(rel_W_star, axis = 0)-1)*l[i]
    return r_i
    
def error_term_invader(species, I):
    k,H,p,l = species
    i,r = [[0,1],[1,0]]
    tM = T*1.01
    t,dt = np.linspace(0,tM,5*int(T),retstep = True)
    t.shape = -1,1,1
    inner_fun = np.exp(l*(t-tM))/(H[r]+I(t))*I.dt(t)/np.log(H[r]+I(tM))
    inner_int = np.cumsum(inner_fun, axis = 0)
    
    return simps(inner_int*np.log(1+I(t)/H[i])/np.log(1+I(t)/H[r])**2, axis = 0)
    
def error_term_resident(species, I):
    k,H,p,l = species
    i,r = [[0,1],[1,0]]
    max_time = T
    t,dt = np.linspace(0,max_time,5*int(T),retstep = True)
    t.shape = -1,1,1
    fun = np.exp(l[r]*(t-max_time))/(H[r]+I(t))*I.dt(t)
    return simps(fun, axis = 0,dx = dt)
    
a = error_term_invader(species, I)       
b = error_term_resident(species, I)
print(a)
print(b)    
# take only species with high balance incoming light
if False:
    k,H,p,l = species
    i,r = [[0,1],[1,0]]
    fit = np.log(p[i]/l[i]/(p[r]/l[r]))
    rel_I_in = np.linspace(-1,1,1000)
    rel_I_in.shape = -1,1
    r_is_uni, r_is_bal = [],[]
    for var in [0,10,20,40,80,85]:
        # randomized arounnd 125
        I_in = var*rel_I_in[:,0]+125
        r_i = ana_r_i(species, I_in)
        """
        plt.figure()
        plt.scatter(*r_i, s = 1, c = fit[0], lw = 0)
        plt.colorbar()
        plt.xlabel(var)
        plt.show()"""
        pos = r_i>0
        coex = pos[0] & pos[1]
        r_is_uni.append(r_i)
        print(np.sum(coex)/coex.size, coex.size)
    
    species = species[..., bal>125]
    fit = fit[...,bal>125]
    bal = bal[bal>125]
    for var in [0,10,20,40,80,85]:
        # optimized around balance
        I_in = var*rel_I_in+bal
        r_i = ana_r_i(species, I_in)
        """
        plt.figure()
        plt.scatter(*r_i, s = 1, c = fit[0], lw = 0)
        plt.colorbar()
        plt.xlabel(var)
        plt.show()"""
        pos = r_i>0
        coex = pos[0] & pos[1]
        print(np.sum(coex)/coex.size, coex.size)
        r_is_bal.append(r_i)
    
    r_is_bal = np.asanyarray(r_is_bal)
    r_is_uni = np.asanyarray(r_is_uni)