# -*- coding: utf-8 -*-
"""
This file is to find the amount of different pigments that exit in nature
To understand the mathematical equations look at \4_analysis, 
Quantification of lightspectrum, pigments.docx
Assumptions: z_m = 1, Sum(\alpha_i)=1
"""

import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import simps, quad
import matplotlib.pyplot as plt

import help_functions_chesson as ches
"""
def pigments_distance(pig1, pig2, alpha_fit = [0.5,2]
                      , I_in = None, exact = False):
    zm = 7.7
    if I_in is None:
        I_in = lambda lam: 40/300 #corresponds to constant light with lux = 40
    integrand1 = lambda lam: (mu1*pig1(lam)-mu2*pig2(lam))\
                                /pig2(lam)*I_in(lam)
    integrand2 = lambda lam: (mu2*pig2(lam)-mu1*pig1(lam))\
                                /pig1(lam)*I_in(lam)
    if exact:
        N1 = pig_equilibrium(mu1, pig1, I_in)
        N2 = pig_equilibrium(mu2, pig2, I_in)

        integrand_1 = lambda lam: integrand1(lam)*(1-np.exp(-N2*zm*pig2(lam)))
        integrand_2 = lambda lam: integrand2(lam)*(1-np.exp(-N1*zm*pig1(lam)))
        return quad(integrand_1,400,700)[0], quad(integrand_2,400,700)[0]
    else:
        
        return quad(integrand1,400,700)[0], quad(integrand2,400,700)[0]
"""
def coexistence(pigs, fit_range):
    equis 

zm = 1
I_in = lambda x:40/300*np.ones(shape = x.shape)
def equilibrium(pigs, fit, ratio = np.linspace(0,1,5)):
    num = len(fit)
    not_found = np.array(num*[True]) #True if equilibriumis found for this setting
    
    x_simps,dx = np.linspace(400,700,101, retstep = True) #uneven number for simpson rule
    absor = lambda ratio, lam: np.einsum('ij,ik->jk',[ratio, 1-ratio],
                            np.array([pig(lam) for pig in pigs]))
    iterator = lambda N_fit, ratio, lam: 40/300*(1-np.exp(-
                        np.einsum('ij,ik->ijk',N_fit,absor(ratio, lam))))
    equi_done = np.infty*np.ones([len(ratio), len(fit)])
    equi = equi_done
    count = 0
    while count < 50 and num>0:

        N_fit = np.einsum('ij,j->ij', equi_done,fit)

        y_simps = iterator(N_fit, ratio, x_simps)
        equi_done = simps(y_simps, dx = dx)
        """save_equi = equi_done.copy()
        equi_done[not_found] = equi
        if count>0:
            not_found = np.logical_and((save_equi-equi_done)/save_equi>0.01,
                        equi_done>0.1)

        num = len(equi_done[not_found])
        x_simps = np.linspace(400,700,101)"""
        count+=1
    return equi_done
    

fit_range = 10**6/0.014*np.linspace(0.5,2,15)
equis = equilibrium([ches.k_green, ches.k_red], fit_range)

plt.plot(fit_range,equis[-1,:])
plt.figure()
plt.plot(fit_range, equis[0,:])

