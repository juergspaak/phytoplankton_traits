# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:46:20 2017

@author: spaakjue
"""
from numpy.random import uniform as uni



def w_photoinhibition(factor=2, return_light = True, return_sol = False):
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
    carbon1 = lambda I: (I*p_max/(a*I**2+b*I+I_k))[0]
    carbon = [carbon0,carbon1]
    
    return species, carbon
