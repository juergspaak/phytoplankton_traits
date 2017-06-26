# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:27:19 2017

@author: Jurg
Generates random environments and checks whether this environment allows existence
"""
import numpy as np
from scipy.optimize import brentq
from numpy.random import uniform as uni
from scipy.integrate import quad

def sat_carbon_par(factor=2,Im = 50, IM = 200):
    """ returns random parameters for the model
    the generated parameters are ensured to survive
    
    the carbon uptake function is saturating (no photoinhibition)
    
    returns species, which contain the different parameters of the species
    species_x = species[:,x]
    carbon = [carbon[0], carbon[1]] the carbon uptake functions
    I_r = [Im, IM] the minimum and maximum incident light"""    
    k = uni(0.004/factor, 0.004*factor,2) # absorption coefficient [m^2/g]
    H = uni(100/factor, 100*factor,2) # halfsaturation for carbon uptake [J/(m^2 s)]
    p_max = uni(9/factor, 9*factor,2) # maximal absorbtion of carbon [s^-1]
    l = uni(0.5/factor, 0.5*factor,2)  # carbon loss [s^-1]
    species = np.array([k,H,p_max,l])
    carbon0 = lambda I: (p_max*I/(H+I))[0]
    carbon1 = lambda I: (p_max*I/(H+I))[1]
    carbon = [carbon0,carbon1]
    return species, carbon,[Im, IM]
    
def photoinhibition_par(factor=2, Im = 100, IM = 500):
    """ returns random parameters for the model
    the generated parameters are ensured to survive
    
    the carbon uptake function suffers from photoinhibition
    
    returns species, which contain the different parameters of the species
    species_x = species[:,x]
    carbon = [carbon[0], carbon[1]] the carbon uptake functions
    I_r = [Im, IM] the minimum and maximum incident light"""
    k = uni(0.002/factor, factor*0.002,2)
    p_max = uni(1/factor, factor*1,2)
    I_k = uni(40/factor, factor*40,2)
    I_opt = uni(100/factor, factor*100,2)
    l = uni(0.5/factor, 0.5*factor,2)  # carbon loss
    species = np.array([k,p_max,I_k, I_opt, l])
    a = I_k/I_opt**2
    b = 1-2*I_k/I_opt
    carbon0 = lambda I: (I*p_max/(a*I**2+b*I+I_k))[0]
    carbon1 = lambda I: (I*p_max/(a*I**2+b*I+I_k))[1]
    carbon = [carbon0,carbon1]
    return species, carbon, [Im, IM]               

def gen_species(parameter_generator):
    """returns two species, for which dominance depends on I_in
    
    parameter_generator shoul be a function that generates 2 species
    gen_species only ensures, that the dominance depends on the incident light
    
    returns:
        same values as parameter_generator"""
    while True: #randomly generate species until they survive
        species, carbon,I_r = parameter_generator()
        Im, IM = I_r
        k = species[0]
        equis = [0,0,0,0]
        equis[0] = equilibrium(Im,species[:,0],carbon[0]) # density, [g/m^2]
        equis[1] = equilibrium(Im,species[:,1],carbon[1]) # density, [g/m^2]
        equis[2] = equilibrium(IM,species[:,0],carbon[0]) # density, [g/m^2]
        equis[3] = equilibrium(IM,species[:,1],carbon[1]) # density, [g/m^2]
        if min(equis)>200 and max(equis)<1e5: #do species survive in monoculture?
            abso0 = k[0]*np.array([equis[0],equis[2]])
            abso1 = k[1]*np.array([equis[1],equis[3]])
            if not ((abso0<abso1).all() or (abso0>abso1).all()):
                #are different species superior at different light regimes?
                if 1.1*Im<find_balance(species, carbon, [Im, IM])<0.9*IM:
                    #is it more or less balanced?
                    return species, carbon, np.array([Im,IM])
                  
def equilibrium(I_in,species,carbon, approx = False):
    """returns the equilibrium of species under light_0
    species and light_0 can either be an array or a number
    light_0 needs the same dimention as species"""
    k,l = species[[0,-1]]
    growth = lambda W: quad((lambda I: carbon(I)/(I*k)),
                            I_in*np.exp(-k*W),I_in)[0]-l*W #should be zero at equilibrium
    
    start_value = quad((lambda I: carbon(I)/(I*k)),0,I_in)[0]/l #start value assumes I_out = 0
    if approx: return start_value #faster, but might be wrong in certain cases
    try:
        return brentq(growth, start_value/1.5, 1.5*start_value, xtol = 1e-3) #finds W^*
    except ValueError:#there is most likely no equilibrium point for this species
        for i in range(20):
            start_value = quad((lambda I: carbon(I)/(I*k)),\
                    I_in*np.exp(-start_value*k),I_in)[0]/l#iterative search for equilibrium
            if np.abs(growth(start_value))<1e-3:
                return start_value
        return np.nan #species seems not to be able to survive

def I_out(I_in, species, carbon):
    """computes the outcoming light at equilibrium"""
    k = species[0]
    return I_in*np.exp(-k*equilibrium(I_in, species, carbon))

def find_balance(species, carbon,I_r):
    """finds the incoming light, at which both species have the same I_out*
    species must be an array containg the parameters of two species, such that
    one species dominates at I_in = 200 and the other at I_in = 50"""
    light_m, light_M = I_r
    I_in = light_m
    I_out = [0,0]
    k,l = species[[0,-1]]
    I_out[0] = I_in*np.exp(-k[0]*equilibrium(I_in,species[:,0],carbon[0]))
    I_out[1] = I_in*np.exp(-k[1]*equilibrium(I_in,species[:,1],carbon[1]))
    direction = I_out[0]>I_out[1] # which species is better at which conditions?
    counter = 0
    tol,diff, I_in = 0.1, 0.5*(light_M-light_m), 0.5* (light_m+light_M)
    while tol<diff and counter <15: #iterates until I_in is found
        I_out[0] = I_in*np.exp(-k[0]*equilibrium(I_in,species[:,0],carbon[0]))
        I_out[1] = I_in*np.exp(-k[1]*equilibrium(I_in,species[:,1],carbon[1]))
        inequal = I_out[0]>I_out[1]
        if inequal == direction: #preparing for next iteration
            light_m = I_in
        else:
            light_M = I_in
        I_in  = 0.5*(light_m+light_M)
        diff = light_M-I_in
        counter +=1
    return I_in

      