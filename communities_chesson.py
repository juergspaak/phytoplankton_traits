# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:27:19 2017

@author: Jurg
Generates random environments and checks whether this environment allows existence
"""
import numpy as np
from scipy.optimize import fsolve
from numpy.random import uniform as uni
import matplotlib.pyplot as plt
from scipy.integrate import quad

def sat_carbon_par(factor=2,Im = 50, IM = 200):
    """ returns random parameters for the model
    the generated parameters are ensured to survive"""
    
    k = uni(0.004/factor, 0.004*factor,2) # absorption coefficient [m^2/g]
    H = uni(100/factor, 100*factor,2) # halfsaturation for carbon uptake [J/(m^2 s)]
    p_max = uni(9/factor, 9*factor,2) # maximal absorbtion of carbon [s^-1]
    l = uni(0.5/factor, 0.5*factor,2)  # carbon loss [s^-1]
    species = np.array([k,H,p_max,l])
    carbon0 = lambda I: (p_max*I/(H+I))[0]
    carbon1 = lambda I: (p_max*I/(H+I))[1]
    carbon = [carbon0,carbon1]
    return species, carbon,Im,IM
    
def photoinhibition_par(factor=2, Im = 100, IM = 1000):
    """ returns random parameters for the model
    the generated parameters are ensured to survive"""
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
    return species, carbon, Im, IM                

def gen_species(parameter_generator):
    while True:
        species, carbon,Im,IM = parameter_generator()
        k = species[0]
        sol0 = equilibrium(Im,species[:,0],carbon[0], True) # density, [g/m^2]
        sol1 = equilibrium(Im,species[:,1],carbon[1], True) # density, [g/m^2]
        if 200<sol0<1e5 and 200<sol1<1e5:
            Sol0 = equilibrium(IM,species[:,0],carbon[0]) # density, [g/m^2]
            Sol1 = equilibrium(IM,species[:,1],carbon[1]) # density, [g/m^2]
            abso0 = k[0]*np.array([sol0,Sol0])
            abso1 = k[1]*np.array([sol1,Sol1])
            if not ((abso0<abso1).all() or (abso0>abso1).all()):
                if 1.1*Im<find_balance(species, carbon, [Im, IM])<0.9*IM:
                    return species, carbon, np.array([Im,IM])
                  
def equilibrium(I_in,species,carbon, approx = False):
    """returns the equilibrium of species under light_0
    species and light_0 can either be an array or a number
    light_0 needs the same dimention as species"""
    k,l = species[[0,-1]]
    growth = lambda W: quad((lambda I: carbon(I)/(I*k)),
                            I_in*np.exp(-k*W),I_in)[0]-l*W #should be zero at equilibrium
    growth_prime = lambda W: [carbon(I_in*np.exp(-k*W))-l] #dgrowth/dW
    start_value = quad((lambda I: carbon(I)/(I*k)),0,I_in)[0]/l #start value assumes I_out = 0
    if approx: return start_value #faster, but might be wrong in certain cases
    equi = fsolve(growth,start_value, fprime = growth_prime) #solves for roots
    return equi[0]

def I_out(I_in, species, carbon):
    k = species[0]
    return I_in*np.exp(-k*equilibrium(I_in, species, carbon))

#species, carbon, I_r = gen_species(photoinhibition_par)

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

      