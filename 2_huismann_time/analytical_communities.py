# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:27:19 2017

@author: Jurg
Generates random environments and checks whether this environment allows existence
"""
import numpy as np

from warnings import warn
from numpy.random import uniform as uni
from string import ascii_lowercase as letters

def generate_com(ncoms = int(1e3), I_r = np.array([50,200])):
    """generates a community"""
    species = find_species(ncoms, I_r)
    balance = find_balance(species, I_r)
    light_range = np.amax([10*np.ones(ncoms),
                   np.amin([balance-I_r[0], I_r[1]-balance],0)],0)
    I_r = np.array([balance-light_range, balance+light_range]).T
    period = np.random.randint(1,200,ncoms)
    return species, period, I_r
        
def find_species(ncoms = 1000, I_r = np.array([50,200])):
    """finds species such that one dominates at I_r[0], the other at I_r[1]"""
    # only about 3.7% fullfill conditions, create to many species
    specs = random_par((int(ncoms/0.02),2)) #randomly generate species
    equis = equilibrium(specs, I_r) #compute their equilibria
    I_out_equiv = np.einsum('is,isl->isl',specs[0],equis) #equivalent to I_out
    #check that species are dominant at different light conditions
    div_dom = np.logical_xor(I_out_equiv[:,0,0]>I_out_equiv[:,1,0],
                             I_out_equiv[:,0,1]>I_out_equiv[:,1,1])
    pot_specs = specs[:,div_dom,:] #choose the ones with different dominance
    if pot_specs.shape[1]<ncoms:
        warn("less species have been returned than asked")
    return pot_specs[:,:ncoms,:]

def random_par(nspecies = (100,2),factor=4, Im = 50):
    """ returns random parameters for the model
    
    Parameters:
        nspecies: int or tuple
            Shape of array that contains number of species
        factor: float
            maximal quotient of two parameters of species
    
    the generated parameters are ensured to survive"""
    factor = factor**0.5
    nspec = int(np.prod(np.array(nspecies)))
    #create twice as many as needed in case some are not able to survive
    k = uni(0.004/factor, 0.004*factor, nspec) # absorption coefficient
    H = uni(100/factor, 100*factor, nspec) #halfsaturation for carbon uptake
    p_max = uni(9/factor, 9*factor, nspec) #maximal absorbtion of carbon
    l = uni(0.5/factor, 0.5*factor, nspec)  # carbon loss
    surv = 80<equilibrium([k,H,p_max,l], Im) #find the ones that survive
    # return the ones that survived, might be less than nspecies, but unlikely
    k = (k[surv]).reshape(nspecies)
    H = (H[surv]).reshape(nspecies)
    p_max = (p_max[surv]).reshape(nspecies)
    l = (l[surv]).reshape(nspecies)
    return np.array([k,H,p_max, l])
                
def equilibrium(specs, light, mode = 'full', 
                l_spe = None, l_lig = None, l_ret = None):
    """returns the equilibrium of species with incoming light `light`
    
    Assumes that I_out = 0
    
    species: array
        An array filled with parameters of (several) species
    light: float or array
        If light is an array the equilibrium density for each species will be
        returned for each light regime
    
    Returns:
        euilibrium: array, shape = (species.shape[1:], len(light))
        Equilibrium density of each species for each incoming lightintensity
    """
    k,H,p_max,l = specs
    if type(light) == float or type(light) == int or mode == 'simple':
        return p_max/(l*k)*np.log(1+light/H)
    elif mode == 'full':
        let = letters[:len(k.shape)] #handles the shape for np.einsum
        fit = p_max/(l*k) #fitness
        #total absorbed light => total energy absorbed
        energy = np.log(1+np.einsum('i,...->...i',light, 1/H))
        return np.einsum(let+','+let+'i->'+let+'i', fit, energy)
    else: 
        fit = p_max/(l*k) #fitness
        
        energy = np.log(1+np.einsum(l_lig+','+l_spe+'->'+l_ret,light, 1/H))
        return np.einsum(l_spe+','+l_ret+'->'+l_ret, fit, energy)

        
def find_balance(specs, I_r = np.array([50,200])):
    """finds the incoming light, at which both species have the same I_out*
    
    does this by intervall searching"""
    k,H,p_max,l = specs
    fit = p_max/l # fitness of species
    
    # function to determine the dominance of the species
    # returns True if species 0 is dominant at I
    dom = lambda I: (1+I/H[:,0])**fit[:,0]/((1+I/H[:,1])**fit[:,1])>1
    dom_Im = dom(I_r[0]) #dominance at minimum light
    
    # decrease I_interest to find equilibrium point?
    decrease_I_inb = lambda I: np.logical_xor(dom_Im, dom(I))
    I_min = I_r[0]*np.ones(len(k)) 
    I_max = I_r[1]*np.ones(len(k))
    #number of iterations needed to reach an error in balance of 0.1
    itera = int(np.log((I_r[1]-I_r[0])/0.1)/np.log(2))+1
    for i in range(itera):
        I_inb = (I_min+I_max)/2 #new guess for I_in balance
        #if decrease, I_max = I_inb, I_min = I_min in next run
        decrease = decrease_I_inb(I_inb)      
        # define new I_min, I_max
        I_max = I_inb*decrease + I_max*np.logical_not(decrease)
        I_min = I_min + I_max - I_inb
    
    return (I_min+I_max)/2 #return average