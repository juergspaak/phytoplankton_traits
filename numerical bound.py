# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:52:15 2017

@author: Jurg
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad, dblquad,odeint
from scipy import interpolate
from timeit import default_timer as timer
import communities_chesson as ches
from communities_chesson import photoinhibition_par, sat_carbon_par


def coex_test(fun):
    species, carbon, I_r = ches.gen_species(fun)
    spec_redo = species
    spec_redo[:,0] = species[:,1]
    spec_redo[:,1] = species[:,0]
    comp1, stor1, r_i_1 = bound_growth(species,carbon, I_r, 35)
    
    carb_redo = [carbon[1], carbon[0]]
    comp2, stor2, r_i_2 = bound_growth(spec_redo, carb_redo, I_r, 35)
    return comp1, stor1, r_i_1, comp2, stor2, r_i_2, species, carbon
  
def dwdt(W,t,species, I_in,carbon):
    """computes the derivative, either for one species or for two
    biomass is the densities of the two species,
    species contains their parameters
    light0 is the incomming light at this given time"""
    dwdt = np.zeros(len(W))
    k,l = species[[0,-1]]
    abso = k[1]*W[1]
    dwdt[0] = k[0]*W[0]/abso*quad(lambda I: carbon[0](I)/(k[0]*I),I_in*math.exp(-abso), I_in)[0]-l[0]*W[0]   
    dwdt[1] = quad(lambda I: carbon[1](I)/(k[1]*I),I_in*math.exp(-abso), I_in)[0]-l[1]*W[1]
    return dwdt
    
def r_i(C,E, species,P,carbon):
    sol = odeint(dwdt, [1,C], [0,P], (species, E, carbon))
    return np.log(sol[1][0]/1)/P #divide by P, to compare different period length

def test_r_i_av(species, P,carbon):
    r_i_val = []
    mini = com.equilibrium(50, species[:,1],carbon[1])
    maxi = com.equilibrium(200, species[:,1],carbon[1])
    for C in np.linspace(mini, maxi, 10):
        print(C)
        for E in np.linspace(50,200,10):
            r_i_val.append(r_i(C,E,species,P,carbon))
    return np.average(r_i_val)
    
def equi_point(species, carbon,I_r):
    E_star = com.find_balance(species, carbon, I_r)
    C_star = com.equilibrium(E_star, species[:,1], carbon[1])
    return E_star, C_star

def bound_growth(species, carbon,I_r ,P):
    start = timer()
    E_star, C_star = equi_point(species, carbon,I_r)
    #print(timer()-start, "tim, equi")
    if C_star <1:
        return None
    curl_C = lambda C: -r_i(C, E_star, species, P,carbon)
    curl_E = lambda E: r_i(C_star, E, species, P,carbon)
    #plotter(curl_C,C_star/2, C_star*2,accuracy = 15 )
    #plt.figure()
    #plotter(curl_E,50, 200,accuracy = 15 )
    Im, IM = I_r
    integrand_C = lambda I: curl_C(com.equilibrium(I, species[:,1], 
                                carbon[1]))/(IM-Im)
    integrand_E = lambda I: curl_E(I)/(IM-Im)
    ave_C = quad(integrand_C, *I_r)[0]
    
    
    
    if math.isnan(ave_C):
        return None
    ave_E = quad(integrand_E, *I_r)[0]
    integrand_stor = lambda I_y, I_x: integrand_C(I_x)*integrand_E(I_y)
    Im_fun = lambda x: Im
    IM_fun = lambda x: IM
    stor = 0#dblquad(integrand_stor, Im, IM, Im_fun, IM_fun)[0]
    gamma =  0#gamma_fun(lambda C,E: r_i(C,E,species, P))(C_star, E_star)
    print(timer()-start, "time end")
    return ave_C, ave_E,ave_E-ave_C
    
def gamma_fun(fun, tol = 0.01):
    def differential(x,y):
        err = 1
        max_eps = 0.1
        while err>tol and max_eps>1e-3:
            eps = np.random.uniform(-max_eps, max_eps, [1000,2])
            delta_eps = [(fun(x+e[0],y+e[1])-fun(x+e[0],y)-fun(x,y+e[1])
                        +fun(x,y))/(e[0]*e[1]) for e in eps]
            err = max(1-np.percentile(delta_eps,90)/np.average(delta_eps),
                      1-np.percentile(delta_eps,10)/np.average(delta_eps))
            print(np.percentile(delta_eps, 5), np.percentile(delta_eps, 95), max_eps)
            eps = np.random.uniform(-max_eps, max_eps, [500,2])
            delta_eps = [(fun(x+e[0],y+e[1])-fun(x+e[0],y)-fun(x,y+e[1])
                        +fun(x,y))/(e[0]*e[1]) for e in eps]
            err = max(1-np.percentile(delta_eps,90)/np.average(delta_eps),
                      1-np.percentile(delta_eps,10)/np.average(delta_eps))
            print(np.percentile(delta_eps, 5), np.percentile(delta_eps, 95), max_eps)
            max_eps /=2
        if max_eps<1e-2:
            e = eps[0]
            print("possibly bad approximation in differentiation", fun, 
                  (fun(x+e[0],y+e[1])-fun(x+e[0],y)-fun(x,y+e[1])+fun(x,y)),
                  (e[0]*e[1]), "av",np.average(delta_eps), max(delta_eps))
        return np.average(delta_eps)
    return differential
    
    
def diff(fun, tol = 0.001):
    def differential(x):
        err = 1
        max_eps = 0.01
        while err>tol and max_eps>1e-10:
            eps = np.random.uniform(-max_eps, max_eps, 10)
            delta_eps = [(fun(x+eps_i)-fun(x))/eps_i for eps_i in eps]
            
            err = 1-min(delta_eps)/max(delta_eps)
            max_eps /=2
        if max_eps<1e-10:
            print("possibly bad approximation in differentiation", fun)
        return np.average(delta_eps)
    return differential
"""    
start = timer()
species, carbon = com.test_species(w_photoinhibition)


a,b,c,d,e,f= bound_growth(species,50, carbon)
print(a,b,e)
end = timer()
print(end-start)"""