# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:27:19 2017

@author: Jurg
Generates random environments and checks whether this environment allows existence
"""
import numpy as np
from scipy.optimize import fsolve
from numpy.random import uniform as uni

def generate_com(light_range = True, light_m = 50, light_M = 200):
    """generates a community"""
    species = find_species(light_m, light_M)
    balance = find_balance(species, light_m, light_M)
    dist = max(10, min(balance-light_m, light_M-balance))
    I_r = [balance-dist, balance+dist]
    period = np.random.randint(5,200)
    if light_range == True:
        return [species, period, balance, I_r]
    else:
        return [species, period, balance]
        
def find_species(light_m = 50, light_M = 200):
    """finds species such that one dominates at I_in = 200, the other at I_in = 50"""
    comp = [True, True]
    while True:
        species = np.array([random_par(2, False),random_par(2, False)]) #random generation of species
        light_outs = np.zeros([2,2])
        for i, light in ([0,[light_m, light_m]],[1,[light_M, light_M]]):
            light_outs[i] = light*np.exp(-np.array(species[:,0])\
                            *equilibrium(light, species))
            
            comp[i] = light_outs[i,0]>light_outs[i,1]
        
        if comp[0]^comp[1]: #check whether they are superior under different light regimes
            return species        

def random_par(factor=10, return_light = True, return_sol = False):
    """ returns random parameters for the model
    the generated parameters are ensured to survive"""
    while True:
        light_0 = uni(100/factor, 100*factor) # light intensity at zero
        k = uni(0.004/factor, 0.004*factor) # absorption coefficient
        H = uni(100/factor, 100*factor) # halfsaturation for carbon uptake
        p_max = uni(9/factor, 9*factor) #maximal absorbtion of carbon
        l = uni(0.5/factor, 0.5*factor)  # carbon loss
        sol = equilibrium(light_0,[k,H,p_max,l])
        if 80<sol<1e5: # about 5% are below 80, i.e. nonexisting, ca 1% above 1e5
            if return_light and return_sol:
                return sol,[light_0,k,H,p_max,l]
            elif return_light and not return_sol:
                return [light_0,k,H,p_max,l]
            elif not return_light and return_sol:
                return sol, [k,H,p_max,l]
            else:
                return [k,H,p_max,l]
                
def equilibrium(light_0,species, approx = False):
    """returns the equilibrium of species under light_0
    species and light_0 can either be an array or a number
    light_0 needs the same dimention as species"""
    if approx:
        return equi_approx(light_0, species)
    array, light_0, species = array_handler(light_0, species)
    equi = np.zeros(len(light_0))
       
    for i in range(len(species)):
        light = light_0[i]
        k,H,p_max,l = species[i]
        start_value = p_max/k*np.log((H+light)/H)/l # start value for iteration assuming that I_out=0
        growth = lambda W: p_max/k*np.log((H+light)/(H+light*np.exp(-k*W)))-l*W #is zero at equilibrium
        equi[i] = fsolve(growth,start_value,xtol=0.1) #solves for roots
    if array:
        return equi
    return equi[0]
    
def equi_approx(light_0, species):
    """returns the equilibrium of species under light_0, assuming I_out = 0
    species and light_0 can either be an array or a number
    light_0 can either be an array or a number"""
    array, light_0, species = array_handler(light_0, species)
    equi = np.zeros(len(light_0))

    for i in range(len(species)):
        light = light_0[i]
        k,H,p_max,l = species[i]
        equi[i] = p_max/(k*l)*np.log(1+light/H)
    if array:
        return equi
    return equi[0]   
                
def find_balance(species, light_m = 50, light_M = 200):
    """finds the incoming light, at which both species have the same I_out*
    species must be an array containg the parameters of two species, such that
    one species dominates at I_in = 200 and the other at I_in = 50"""
    light_outs = [0,0]
    light = np.array([light_m, light_m])
    light_outs = light*np.exp(-np.array(species[:,0])*equilibrium(light, species))
    direction = light_outs[0]>light_outs[1] # which species is better at which conditions?
    counter = 0
    tol,diff, light = 0.1, 0.5*(light_M-light_m), 0.5* (light_m+light_M)
    while tol<diff and counter <10: #iterates until I_in is found
        light_outs = light*np.exp(-np.array(species[:,0])*\
                                  equilibrium([light,light], species))
        inequal = light_outs[0]>light_outs[1]
        if inequal == direction: #preparing for next iteration
            light_m = light
        else:
            light_M = light
        light  = 0.5*(light_m+light_M)
        diff = light_M-light
        counter +=1
    return light


def array_handler(*args, depth = 1):
    try:
        a = args[0]
        for i in range(depth):
            a = a[0]
        return (True, *args)
    except (TypeError, IndexError):
        returns = []
        for arg in args:
            returns.append([arg])
        return (False,*tuple(returns))