# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:27:19 2017

@author: Jurg
Generates random environments and checks whether this environment 
allows existence

This file is equivalent to numerical_communities. Functions with the same
name serve the same purpose. The functions in this file serve for analytical_r_i

`gen_species` returns two random species

`random_par`: rand. generate two species with saturating carbon uptake
    functions
    
`equilibrium` computes the equilibrium density of the species

`find_balance`: Find the incoming light intensity where both species would be
    able to coexist (instable)


"""
import autograd.numpy as np

from warnings import warn
from numpy.random import uniform as uni

        
def gen_species(num = 1000, I_r = np.array([50,200])):
    """returns two species, for which dominance depends on I_in
    
    parameters:
        num: int, number of communities to construct
        I_r: array, minimal and maximal incoming light
    
    Returns:
        same values as random_par"""
    # only about 3.7% fullfill conditions, create to many species
    species = random_par((2,int(num/0.02))) #randomly generate species
    
    equis = equilibrium(species, I_r) #compute their equilibria
    #equivalent to I_out
    I_out_equiv = species[0].reshape((1,)+species[0].shape)*equis
    #check that species are dominant at different light conditions
    div_dom = np.logical_xor(I_out_equiv[0,0]>I_out_equiv[0,1],
                             I_out_equiv[1,0]>I_out_equiv[1,1])
    pot_species = species[:,:, div_dom] #choose the ones with different dominance
    if pot_species.shape[-1]<num:
        warn("less species have been returned than asked")
    return pot_species[:,:,:num].copy() #copy to empty memory

def random_par(nspecies = (2,100),factor=100, Im = 50):
    """ returns random parameters for the model
    
    Parameters:
        nspecies: int or tuple
            Shape of array that contains number of species
        factor: float
            maximal quotient of two parameters of species
    
    Returns:
        species = np.array([k,H,p_max, l])
            absorption coefficient, halfsaturating constant of carbon uptake,
            maxiaml carbon uptake, specific loss rate"""
    factor = factor**0.5
    nspec = int(np.prod(np.array(nspecies)))
    #create twice as many as needed in case some are not able to survive
    k = uni(0.004/factor, 0.004*factor, 2*nspec) # absorption coefficient
    H = uni(100/factor, 100*factor, 2*nspec) #halfsaturation for carbon uptake
    p_max = uni(9/factor, 9*factor, 2*nspec) #maximal absorbtion of carbon
    l = uni(0.5/factor, 0.5*factor, 2*nspec)  # carbon loss
    surv = 80<equilibrium([k,H,p_max,l], Im) #find the ones that survive
    # return the ones that survived, might be less than nspecies, but unlikely
    k = (k[surv][:nspec]).reshape(nspecies)
    H = (H[surv][:nspec]).reshape(nspecies)
    p_max = (p_max[surv][:nspec]).reshape(nspecies)
    l = (l[surv][:nspec]).reshape(nspecies)
    return np.array([k,H,p_max, l])
                
def equilibrium(species, I_in, mode = None):
    """returns the equilibrium of species with incoming light `light`
    
    Assumes that I_out = 0
    
    Parameters:
        species: return value of random_par
        I_in: float or array
            If light is an array the equilibrium density for each species 
            will be returned for each light regime
        mode: None, 'simple, 'partial' or 'full'
            Determines the shape of return value
    
    Returns:
        equi: array
            equilibrium of species at I_in
    """
    k,H,p_max,l = species
    fit = p_max/(l*k)
    try:
        I_inv = I_in.view()  # to not change the shape of I_in
    except AttributeError:
        I_inv = I_in
    if isinstance(I_in,(int, float, np.int32, np.float)):
        pass
    elif mode == 'full' or (I_in.ndim==1 and len(I_in)!=species.shape[-1]):
        # equilibrium of each species[...,i] for each entry of I_in[j]
        I_inv.shape = (-1,)+species[0].ndim*(1,)
    elif mode == 'simple' or (I_in.ndim==1 and len(I_in)==species.shape[-1]):
        # equilibrium of each species[...,i] for the same entry of I_in[i]
        pass
    elif mode=='partial' or (I_in.ndim==2 and I_in.shape[-1]==species.shape[-1]):
        # combination of 'simple' and 'full'. Compute the equilibria of each 
        # species[...,i] for each entry in I_in[:,i]
        I_inv.shape = len(I_in),1,-1
    else:
        raise ValueError("""I_in must be a np.array (of dimension 1 or 2) or a 
        scalar, if possible please specify `mode`.""")
        
    return fit*np.log(1+I_inv/H)    

        
def find_balance(species, I_r = np.array([50,200])):
    """finds the incoming light, at which both species have the same I_out*
    
    Parameters:
        species: return values of random_par
    
    Returns:
        I_in: array
        Incoming light at which nonstable coexistence occurs"""
    k,H,p_max,l = species
    fit = p_max/l # fitness of species
    
    # function to determine the dominance of the species
    # returns True if species 0 is dominant at I
    dom = lambda I: (1+I/H[0])**fit[0]/((1+I/H[1])**fit[1])>1
    dom_Im = dom(I_r[0]) #dominance at minimum light
    
    # decrease I_interest to find equilibrium point?
    decrease_I_inb = lambda I: np.logical_xor(dom_Im, dom(I))
    I_min = I_r[0]*np.ones(k.shape[-1]) 
    I_max = I_r[1]*np.ones(k.shape[-1])
    # number of iterations needed to reach an error in balance of 0.1
    itera = int(np.log((I_r[1]-I_r[0])/0.1)/np.log(2))+1
    for i in range(itera):
        I_inb = (I_min+I_max)/2 #new guess for I_in balance
        #if decrease, I_max = I_inb, I_min = I_min in next run
        decrease = decrease_I_inb(I_inb)      
        # define new I_min, I_max
        I_max = I_inb*decrease + I_max*np.logical_not(decrease)
        I_min = I_min + I_max - I_inb
    
    return (I_min+I_max)/2 #return average