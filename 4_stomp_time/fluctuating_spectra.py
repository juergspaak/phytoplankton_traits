"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Contains functions that generate coexisting communities with changing incoming
light
"""
import sys
sys.path.append("../3_different_pigments")
import load_pigments as lp
from load_pigments import lambs, dlam
import multispecies_functions as mf
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from ode_solving import own_ode
from scipy.integrate import simps

import numpy as np

def gen_com(n_com = 1000):
    # generates communities
    r_pig, r_spec, r_pig_spec, fac = 5, 10, 3, 3 # richness etc.
    k_spec, alpha = mf.spectrum_species(lp.real, r_pig,r_spec,n_com,r_pig_spec)
     # specific photosynthetic efficiency and loss rate
    phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
    return np.array([phi,l]), k_spec, alpha
                    
def I_in_t(I_in1, I_in2, period):
    # returns light at given time t, fluctuates between I_in1, I_in2
    def fun(t):
        t_rel = (t%period)/period
        part_1 = 2*np.abs(t_rel-0.5)
        return part_1*I_in1+(1-part_1)*I_in2
    return fun
"""
###############################################################################
# find potentially interesting communities

# light regimes
I_ins = [mf.I_in_def(40/300, 650,50), mf.I_in_def(40/300, 450,50)]
         
# generate species and communities
par, k_spec, alpha = gen_com(1000)
phi,l = par

# compute the equilibria densities for the different light regimes
equi = np.empty((len(I_ins),) + phi.shape)
unfixed = np.empty((len(I_ins),phi.shape[-1]))
for i,I_in in list(enumerate(I_ins)):
    equi[i], unfixed[i] = mf.multispecies_equi(phi/l, k_spec, I_in)

# consider only communities, where algorithm found equilibria (all regimes)
fixed = np.logical_not(np.sum(unfixed, axis = 0))

# find cases, equilibria species change
surv = equi>0 # species that survived
# XOR(present in one, present in all)
change_dom = np.logical_xor(np.sum(surv, axis = 0), np.prod(surv, axis = 0))
change_dom = change_dom.sum(axis = 0) # at least one species changes

#throw away uninteressting communities (i.e. not fixed, no change of dominance)
interesting = np.logical_and(change_dom, fixed)

# throw away communities, that have not been fixed/have no change in dominance
phi = phi[...,interesting]
l = l[..., interesting]
k_spec = k_spec[...,interesting]
equi = equi[..., interesting]

# set 0 all species that did not survive in any of the cases
dead = np.sum(surv[...,interesting], axis = 0)==0
phi[dead] = 0
l[dead] = 1 # to aboid division by 0
k_spec[:,dead] = 0

# maximal richness over all environments in one community
max_spec = np.amax(np.sum(np.sum(equi>0, axis = 0)>0, axis = 0))
# sort them accordingly to throw rest away
com_ax = np.arange(equi.shape[-1])
spec_sort = np.argsort(np.amax(equi,axis = 0), axis = 0)[-max_spec:]
equi = equi[np.arange(2).reshape(-1,1,1),spec_sort, com_ax]
phi = phi[spec_sort, com_ax]         
l = l[spec_sort, com_ax]
k_spec = k_spec[np.arange(len(lambs)).reshape(-1,1,1),spec_sort, com_ax]
"""

# take maximum densitiy over all lights for the starting density of the species
start_dens = np.amax(equi, axis = 0)

def multi_growth(N,t,I_in, k_spec, phi,l):
    # sum(N_j*k_j(lambda))
    tot_abs = np.einsum("sc,lsc->lc", N, k_spec)[:,np.newaxis]
    # growth part
    growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                       *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
    return N*(growth-l)

period = 10
I_in_ref = I_in_t(*I_ins, period)
a = multi_growth(start_dens,1,I_in_ref,k_spec, phi, l)

# compute 100 periods, 10 timepoints per period
time = np.linspace(0,100*period,period*100*10)

sol = own_ode(multi_growth,start_dens, time[[0,-1]], args=(I_in_ref, k_spec, phi,l),
    steps = len(time))

###############################################################################
# determine extinction prone species
av_end = np.average(sol[-10:], axis = 0) # average over one period at end
av_start = np.average(sol[-110:-100], axis = 0) # after finding "equilibrium"

rel_diff = (av_end-av_start)/av_start # relative difference in start and end
ex_id = np.nanargmin(ex, axis = 0) # species with most difference
ex_com = ex[ex_id, np.arange(ex.shape[-1])]>0.01
print(rel_diff[ex_id, np.arange(ex.shape[-1])][:10])

def invasion(res_dens, k_inv, k_spec, I_int, phi_i, l_i):
    tot_abs = np.einsum("tsc,lsc->tlc", res_dens, k_spec)[:,:,np.newaxis]
    # growth part
    growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                       *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
    return N*(growth-l)





if False:
    # test whether solving ode was correct
    def single_growth(N,t,I_in,k_spec, phi, l):
        tot_abs = np.sum(N*k_spec,axis =1)[:,np.newaxis]
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1),dx = dlam, axis = 0)
        return N*(growth-l)
    i = 10
    start = timer()
    sol2 = odeint(single_growth, start_dens[:,i], time, 
                  args = (I_in_ref, k_spec[...,i], phi[:,i],l[:,i]) )
    print(timer()-start)
    plt.plot(time, sol[:,:,i] )
    plt.plot(time, sol2)
