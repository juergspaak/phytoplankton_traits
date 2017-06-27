"""
@author: J.W.Spaak

Contains functions to compute the boundary growth rate of species

Species: Method to create species and compute their boundary growth rates

`bound_growth` function to compute the boundary growth rate numerically

`equi_point` function to compute the instable equilibrium

`r_i` numerically computes the growth rate in one period

`dwdt`right hand side of the differential equation

`test_r_i_av` alternative (randomize) way to compute r_i
"""

import numpy as np
import math
from scipy.integrate import quad,odeint
import communities_chesson as ches

class Species:
    """generates a pair of species
    
    The species are generated by the function while running
    parameter_gen should be a function to generate species
    
    Example:
        spec = Species(ches.photoinhibition_par) #create species
        spec.new_period(5) #compute the boundary growthrate of period 5
        spec.r_i #check the boundary growth rate
        
    """
    def __init__(self, parameter_gen):
        """computes the species parameters"""
        species, carbon, I_r = ches.gen_species(parameter_gen)
        self.species = species #species parameters
        self.carbon = carbon #carbonuptake functions, [carbon1(I), carbon2(I)]
        self.I_r = I_r #range of incomming light intensity
        self.P = np.array([]) #to save different period lengths
        
        spec_redo = species.copy() 
        spec_redo[:,0] = species[:,1].copy()
        spec_redo[:,1] = species[:,0].copy()
        carb_redo = [carbon[1], carbon[0]]
                     
        self._redo = [spec_redo, carb_redo] #to compute r_2
        self.comp = np.array([]) #to save competition
        self.envi = np.array([]) #to save DeltaE
        self.r_i = np.array([]) #to save boundary grwoth rates
        
    def new_period(self, P = None):
        """computes the boundary growth rate for a new period length
        
        if P is None it will be randomly generated"""
        if P is None:
            P = np.random.uniform(5,50)
        self.P = np.append(self.P,P) #append P
        
        #compute the parameters
        comp1, envi1, r_i_1 = bound_growth(self.species,self.carbon,
                                           self.I_r, self.P[-1])
        comp2, envi2, r_i_2 = bound_growth(self._redo[0], self._redo[1],
                                           self.I_r, self.P[-1])
        #store the computed parameters
        self.comp = np.append(self.comp,np.array([comp1,comp2]))
        self.envi = np.append(self.envi,np.array([envi1,envi2]))
        self.r_i = np.append(self.r_i,np.array([r_i_1, r_i_2]))
    
    def equi(self,light,i):
        """computes the equilibrium of species i at light intensity light"""
        return ches.equilibrium(light, self.species[:,i], self.carbon[i])

def bound_growth(species, carbon,I_r ,P):
    """computes av(r_i) for species
    
    species, carbon, I_r should be outputs of gen_species
    I_r should be the range in which I_in fluctuates
    
    returns deltaC, delta E, r_i = deltaE-deltaC
    
    delta E is assumed to be very small"""
    E_star, C_star = equi_point(species, carbon,I_r)
    if C_star <1:
        print("printed by bound_growth",E_star, C_star)

    curl_C = lambda C: -r_i(C, E_star, species, P,carbon)
    curl_E = lambda E: r_i(C_star, E, species, P,carbon)
    
    Im, IM = I_r
    integrand_C = lambda I: curl_C(ches.equilibrium(I, species[:,1], 
                                carbon[1]))/(IM-Im)
    integrand_E = lambda I: curl_E(I)/(IM-Im)
    ave_C = quad(integrand_C, *I_r)[0]
    """
    if math.isnan(ave_C):
        return None"""
    ave_E = quad(integrand_E, *I_r)[0]
    integrand_stor = lambda I_y, I_x: integrand_C(I_x)*integrand_E(I_y)
    Im_fun = lambda x: Im
    IM_fun = lambda x: IM
    stor = 0#dblquad(integrand_stor, Im, IM, Im_fun, IM_fun)[0]
    gamma =  0#gamma_fun(lambda C,E: r_i(C,E,species, P))(C_star, E_star)
    return ave_C, ave_E,ave_E-ave_C
    
def equi_point(species, carbon,I_r):
    """computes the incoming light for unstable coexistence
    
    returns E_star, C_star
    
    species, carbon, I_r should be outputs of gen_species"""
    E_star = ches.find_balance(species, carbon, I_r)
    C_star = ches.equilibrium(E_star, species[:,1], carbon[1])
    return E_star, C_star

def r_i(C,E, species,P,carbon):
    """computes r_i for the species[:,0] assuming species[:,1] at equilibrium
    
    does this by solving the differential equations
    C should be the density of the resident species, E the incomming light
    P the period length, carbon = [carbon[0],carbon[1]] the carbon uptake 
    functions of both species
    
    returns r_i for this period"""
    sol = odeint(dwdt, [1,C], [0,P], (species, E, carbon))
    return np.log(sol[1][0]/1)/P #divide by P, to compare different period length
        
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

def test_r_i_av(species, P,carbon):
    """computes av(r_i), the average boundary_growth rate
    
    species, carbon should be the species parameters, computed by gen_species
    P is the period length
    Averaging is done by taking the outcomes of different starting C and E
    """
    r_i_val = []
    mini = ches.equilibrium(50, species[:,1],carbon[1])
    maxi = ches.equilibrium(200, species[:,1],carbon[1])
    for C in np.linspace(mini, maxi, 10):
        print(C)
        for E in np.linspace(50,200,10):
            r_i_val.append(r_i(C,E,species,P,carbon))
    return np.average(r_i_val)