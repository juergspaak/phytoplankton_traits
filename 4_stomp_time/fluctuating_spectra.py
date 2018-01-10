"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Contains functions that generate coexisting communities with changing incoming
light
"""
import numpy as np
from scipy.integrate import simps

import sys
sys.path.append("../3_different_pigments")
import load_pigments as lp
from load_pigments import lambs, dlam
import multispecies_functions as mf

from ode_solving import own_ode

def gen_com(r_pig, r_spec, r_pig_spec, fac,pigs = "real", n_com = 1000):
    if pigs == "rand":
        k_spec, alpha = mf.spectrum_species(lp.rand, 
                                            r_pig,r_spec,n_com,r_pig_spec)
    else:
        k_spec, alpha = mf.spectrum_species(lp.real, 
                                            r_pig,r_spec,n_com,r_pig_spec)
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

I_in_ref = I_in_t(mf.I_in_def(40/300,450,50), mf.I_in_def(40/300,650,50),10)
   
def fluctuating_richness(r_pig = 5, r_spec = 10, r_pig_spec = 3,n_com = 1000,
    fac = 3, l_period = 10, pigs = "real", I_in = I_in_ref,t_const = [0,0.5]):
    ###########################################################################
    # find potentially interesting communities
             
    # generate species and communities
    par, k_spec, alpha = gen_com(r_pig, r_spec, r_pig_spec, fac,pigs, n_com)
    phi,l = par
    
    # compute the equilibria densities for the different light regimes
    equi = np.empty((len(t_const),) + phi.shape)
    unfixed = np.empty((len(t_const),phi.shape[-1]))

    for i,t in list(enumerate(t_const)):
        equi[i], unfixed[i] = mf.multispecies_equi(phi/l, k_spec, 
                            I_in(t*l_period))
    # consider only communities, where algorithm found equilibria (all regimes)
    fixed = np.logical_not(np.sum(unfixed, axis = 0))
    equi = equi[..., fixed]
    phi = phi[:, fixed]
    l = l[:, fixed]
    k_spec = k_spec[..., fixed]

    # richness in constant lights
    richness_const = np.sum(equi>0, axis = 1)
    
    # find cases, equilibria species change
    surv = equi>0 # species that survived
    # XOR(present in one, present in all)
    change_dom = np.logical_xor(np.sum(surv, axis = 0), 
                                np.prod(surv, axis = 0))
    change_dom = change_dom.sum(axis = 0)>0 # at least one species changes
    
    if change_dom.sum()==0: # fluctuation is unimportant/didn't change anything
        # find number of coexisting species through time
        richness_fluc,richness_const_max = richness_const[:2]
    
        ret_mat = np.array([[(richness==i+1).sum() for i in range(10)] for 
            richness in [*richness_const, richness_const_max, richness_fluc]])
        return ret_mat/ret_mat.sum(axis = 1).reshape(-1,1) # normalize   
        
    # throw away communities, that have no change in dominance
    phi = phi[...,change_dom]
    l = l[..., change_dom]
    k_spec = k_spec[...,change_dom]
    equi = equi[..., change_dom]
    
    # set 0 all species that did not survive in any of the cases
    dead = np.sum(surv[...,change_dom], axis = 0)==0
    phi[dead] = 0
    l[dead] = 1 # to aboid division by 0
    k_spec[:,dead] = 0
    # maximal richness over all environments in one community
    max_spec = np.amax(np.sum(equi>0, axis = 1))
    # sort them accordingly to throw rest away
    com_ax = np.arange(equi.shape[-1])
    spec_sort = np.argsort(np.amax(equi,axis = 0), axis = 0)[-max_spec:]
    phi = phi[spec_sort, com_ax]         
    l = l[spec_sort, com_ax]
    equi = equi[np.arange(len(t_const)).reshape(-1,1,1),spec_sort, com_ax]
    k_spec = k_spec[np.arange(len(lambs)).reshape(-1,1,1),spec_sort, com_ax]
       
    ###########################################################################
    # take maximum densitiy over all lights for the starting density
    start_dens = np.amax(equi, axis = 0)

    def multi_growth(N,t,I_in, k_spec, phi,l):
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = np.einsum("sc,lsc->lc", N, k_spec)[:,np.newaxis]
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
        return N*(growth-l)
    
    n_period = 100 # number of periods to simulate
    
    undone = np.arange(phi.shape[-1])
    # compute 100 periods, 10 timepoints per period
    time = np.linspace(0,l_period*n_period,n_period*10)
    phit,lt,k_spect = phi.copy(), l.copy(), k_spec.copy()
    # to save the solutions found
    sols = np.empty((10,)+phi.shape)
    
    # simulate densities
    counter = 1

    while len(undone)>0 and counter <1000:
        sol = own_ode(multi_growth,start_dens, time[[0,-1]], 
                      args=(I_in, k_spect, phit,lt),steps = len(time))
        
        # determine change in densities, av at end and after finding equilibria
        av_end = np.average(sol[-10:], axis = 0) 
        av_start = np.average(sol[-110:-100], axis = 0) 
        
        # relative difference in start and end
        rel_diff = np.nanmax(np.abs((av_end-av_start)/av_start),axis = 0)
        # communities that still change "a lot"
        unfixed = rel_diff>0.005
        
        # save equilibria found
        sols[...,undone] = sol[-10:]
        
        # select next communities
        undone = undone[unfixed]
        phit,lt,k_spect = phi[:, undone], l[:, undone], k_spec[...,undone]
        start_dens = sol[-1,:,unfixed].T
        # remove very rare species
        start_dens[start_dens<start_dens.sum(axis = 0)/5000] = 0
        counter += 1             
    
    ###########################################################################
    # preparing return values
                  
    # find number of coexisting species through time
    richness_fluc = np.sum(sols[-1]>0,axis = 0)
    # add the cases where fluctuations didn't change anything
    richness_fluc = np.append(richness_const[0,~change_dom],richness_fluc)
    # take the maximal number of coexisting species in each community
    richness_const_max = np.amax(richness_const, axis = 0)
    
    ret_mat = np.array([[(richness==i+1).sum() for i in range(10)] for richness 
                    in [*richness_const, richness_const_max, richness_fluc]])
    return ret_mat/ret_mat.sum(axis = 1).reshape(-1,1) # normalize


