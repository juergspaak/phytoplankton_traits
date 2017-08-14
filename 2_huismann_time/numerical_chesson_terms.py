"""
@author: J.W.Spaak

Contains functions to compute the boundary growth rate of species

Species: Method to create species and compute their boundary growth rates

Most functions are similar to numerical_r_i, adapt these to include chesson terms
"""

import numpy as np
from scipy.integrate import simps,odeint
import communities_chesson as ches

def bound_growth(spec, carbon,I_r ,P):
    """computes av(r_i) for species
    
    species, carbon, I_r should be outputs of gen_species
    I_r should be the range in which I_in fluctuates
    
    returns ave_C, ave_E, r_i = ave_E-ave_C
    
    ave_E is assumed to be very small"""
    if I_r.ndim == 1:
        I_r = np.ones((1,spec.shape[-1]))*I_r[:,np.newaxis]
                       
    # compute unstable equilibrium point
    E_star, C_star = equi_point(spec, carbon, I_r)
    
    Im, IM = I_r #light range
    # density in dependence of incoming light
    C = lambda I: ches.equilibrium(spec, carbon, I, 'simple')
    # growth in one period
    curl_C = lambda I: -r_i(C(I), E_star, spec, P,carbon)
    curl_E = lambda I: r_i(C_star, I, spec, P,carbon)

    acc_rel_I = 21 # number used for simpson integration
    # uniform distribution in 0,1, used for integration
    rel_I,dx = np.linspace(1e-10,1,acc_rel_I,retstep = True)
    rel_I.shape = 1,-1
    
    # Effective light used for integration
    I_eff = np.expand_dims(IM-Im,-1)*rel_I+np.expand_dims(Im,-1)
    # x-values used for simpson rule
    simps_C = np.zeros((acc_rel_I,)+spec.shape[1:])
    simps_E = np.zeros((acc_rel_I,)+spec.shape[1:])
    for i in range(acc_rel_I):
        print("please read comments on line 48ff")
        """ this lines compute the boundary growth rate, but it has to be multiplied
        with the partial differential: r_i(E,C) = dr/dC*(C-C^*)..."""
        simps_C[i] = curl_C(I_eff[:,i])
        simps_E[i] = curl_E(I_eff[:,i])
    
    # integrate, divide by light range because of variable transformation   
    ave_C = simps(simps_C, dx = dx, axis = 0)/(IM-Im)
    ave_E = simps(simps_E, dx = dx, axis = 0)/(IM-Im)
    """
    integrand_stor = lambda I_y, I_x: integrand_C(I_x)*integrand_E(I_y)
    Im_fun = lambda x: Im
    IM_fun = lambda x: IM
    stor = 0#dblquad(integrand_stor, Im, IM, Im_fun, IM_fun)[0]
    gamma =  0#gamma_fun(lambda C,E: r_i(C,E,species, P))(C_star, E_star)"""
    return ave_C, ave_E,ave_E-ave_C


def equi_point(species, carbon,I_r):
    """computes the incoming light for unstable coexistence
    
    returns E_star, C_star
    
    species, carbon, I_r should be outputs of gen_species"""
    E_star = ches.find_balance(species, carbon, I_r)
    C_star = ches.equilibrium(species, carbon, E_star, 'simple')
    if (C_star <1).any(): #avoid errors
        print("printed by equi_point",np.amin( C_star))
    return E_star, C_star

def r_i(C,E, species,P,carbon):
    """computes r_i for the species[:,0] assuming species[:,1] at equilibrium
    
    does this by solving the differential equations
    C should be the density of the resident species, E the incomming light
    P the period length, carbon = [carbon[0],carbon[1]] the carbon uptake 
    functions of both species
    
    returns r_i for this period"""
    # first axis is for invader/resident, second for the two species
    start_dens = np.array([np.ones(C.shape),C])
    #compute the growth of the two species
    sol = odeint(dwdt, start_dens.reshape(-1), [0,P],args = (species, E, carbon))
    sol.shape =  2,2,2,-1 #odeint only alows 1-dim arrays        
    #divide by P, to compare different period length
    return np.log(sol[1,0,[1,0]]/1)/P #return in different order, because species are swapped


        
def dwdt(W,t,species, I_in,carbon):
    """computes the derivative, either for one species or for two
    biomass is the densities of the two species,
    species contains their parameters
    light0 is the incomming light at this given time"""
    if W.ndim == 1: #reshape into usable shape
        W = W.reshape(2,2,-1)
    k,l = species[[0,-1]]
    # relative absorption: (k*W)[invader]/(k*W)[resident]
    rel_abso = np.ones(W.shape)
    res_abso = k*W[1]
    rel_abso[1,0] = W[0,0]*k[1]/(k[0]*W[1,0])
    rel_abso[0,1] = W[0,1]*k[0]/(k[1]*W[1,1])
    
    
        
    # relative light, needed for integration
    rel_I,dx = np.linspace(1e-10,1,21,retstep = True)
    rel_I = rel_I.reshape((1,1,-1))
    
    I_out = I_in*np.exp(-res_abso) # outcoming light for species
    I_eff = np.expand_dims(I_in-I_out,-1)*rel_I+np.expand_dims(I_out,-1)
    
    # carbon uptake of the species at I_eff
    carb_up = carbon(species,I_eff,'special')/(k[:,None,:,None]*I_eff)
    
    #integrate carbonuptake, integral computet with linear trans.
    growth = simps(carb_up,dx = dx,axis=-1)\
                           *(I_in-I_out)
    
    # loss of species
    loss = np.array([[l[1],l[0]],[l[0],l[1]]])*W
    reloss = loss.copy()
    reloss[0,0], reloss[1,0] = loss[1,0], loss[0,0]
    new = rel_abso*growth-reloss #rearanging to meet requirements
    dwdt = new.copy()
    dwdt[1,0] = new[0,0]
    dwdt[0,0] = new[1,0]
    dwdt[0,1] = new[0,1]
    return dwdt.reshape(-1)
    
if False: #generate species and compute bound growth
    spec, carb, I_r = ches.gen_species(ches.sat_carbon_par, num = 5000)
    a,b,c = bound_growth(spec, carb, I_r,50)    
    
if False: # check r_i is doing a good job
    I = np.random.uniform(50,200, spec.shape[-1])
    r_i(ches.equilibrium(spec, carb, I), I,spec, 10, carb, True)
    

