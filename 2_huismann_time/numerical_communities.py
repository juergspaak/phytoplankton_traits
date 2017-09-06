"""
@author: J.W.Spaak

This file is equivalent to analytical_communities. Functions with the same
name serve the same purpose. The functions in this file serve for numerical_r_i

Generates random species, that can be used for numerical_r_i

`gen_species` returns two random species

`sat_carbon_par`: rand. generate two species with saturating carbon uptake
    functions
`photoinhibition_par`: rand. generate two species with carbon uptake
    functions that suffer from photoinhibition
    
`equilibrium` computes the equilibrium density of the species

`find_balance`: Find the incoming light intensity where both species would be
    able to coexist (instable)

"""
import numpy as np
from numpy.random import uniform as uni
from scipy.integrate import simps

###############################
# equivalent to gen_species in analytical_communities
###############################

def gen_species(parameter_generator, num = 1000):
    """returns two species, for which dominance depends on I_in
    
    Parameters:
        parameter_generator: sat_carbon_par or photoinhibition_par
        num: int, number of communities to construct. Will only return species
            that might coexist. This may be less than num
    
    returns:
        same values as parameter_generator"""
    species, carb, I_r = parameter_generator(num)
    def carbon(species, I, mode = 'full'):
        if isinstance(I,(int, float)) or mode == 'simple':
            pass
        elif mode == 'full':
            I_shape = I.ndim*(1,)
            species = [par.reshape(par.shape+I_shape) for par in species]
            I = I.reshape(species[0].ndim*(1,)+I.shape)
        elif mode == 'partial':
            I_shape = (I.ndim-1)*(1,)
            I = I.reshape((1,)+I.shape[::-1])
            species = [par.reshape(par.shape+I_shape) for par in species]
        elif mode == 'generating':
            species = np.expand_dims(species,-1)
        elif mode == 'special':
            species = species.reshape(species.shape+(1,1)).swapaxes(-3,-2)
            I = np.rollaxis(I,1,2)
            I = I.reshape((1,)+I.shape)
        return carb(species, I)
    k = species[0]

    # compute absorption of both species (equivalent to I_out)
    equis = equilibrium(species, carbon, I_r, 'full')
    # dominance of species
    dominance = k[0]*equis[:,0]>k[1]*equis[:,1] 
    # balanced if dominance changes
    balanced = np.logical_xor(dominance[0], dominance[1])
    # species must have positive aboundancies
    survive = np.amin(equis, axis = 1)>0
    # species that are balanced and survive
    good = np.logical_and(balanced, np.logical_and(survive[0], survive[1]))
    return species[:,:,good], carbon, I_r

###############################
# equivalent to random_par in analytical_communities
###############################
    
def sat_carbon_par(num = 1000, factor=4, I_r = np.array([50,200])):
    """ returns random parameters for the model p(I) = p_max*I/(I+H)
    
    Paramters:
        num: int, number of communities to generate
        factor: maximal quotient two species have in one parameter
        I_r: array, minimal and maximal incident light
    
    Returns:
        species = np.array([k,H,p_max, l])
            absorption coefficient, halfsaturating constant of carbon uptake,
            maxiaml carbon uptake, specific loss rate
        carbon: callable
            carbon(I) = p_max*I/(I+H)
        I_r: array, minimal and maximal incident light
    """
    fac = np.sqrt(factor)
    k = uni(0.004/fac, 0.004*fac,(2,num)) # absorption coefficient [m^2/g]
    H = uni(100/fac, 100*fac,(2,num)) # halfsaturation for carbon uptake [J/(m^2 s)]
    p_max = uni(9/fac, 9*fac,(2,num)) # maximal absorbtion of carbon [s^-1]
    l = uni(0.5/fac, 0.5*fac,(2,num))  # carbon loss [s^-1]
    species = np.array([k,H,p_max,l])
    def carbon(species, I):
        return species[2]*I/(species[1]+I)
    return species, carbon, I_r
    
def photoinhibition_par(num = 1000, factor=4, I_r = np.array([100.,1000.])):
    """ returns random parameters for the model with photoinhibition
    
    Paramters:
        num: int, number of communities to generate
        factor: maximal quotient two species have in one parameter
        I_r: array, minimal and maximal incident light
    
    Returns:
        species = np.array([k,p_max,I_k, I_opt, l])
            absorption coefficient,  maximal carbon uptake, I_k = dp/dI|I=0,
            optimal incicdent light, specific loss rate
        carbon: callable
            carbon(I) = carbon uptake of species
        I_r: array, minimal and maximal incident light
    """
    fac = np.sqrt(factor)
    k = uni(0.002/fac, fac*0.002,(2,num)) #values from gerla et al 2011
    p_max = uni(5/fac, fac*5,(2,num))
    I_k = uni(40/fac, fac*40,(2,num))
    I_opt = uni(200/fac, fac*200,(2,num))
    l = uni(0.5/fac, 0.5*fac,(2,num))  # carbon loss
    species = np.array([k,p_max,I_k, I_opt, l])
    def carbon(species, I):
        a = species[2]/species[3]**2
        b = 1-2*species[2]/species[3]
        return species[1]*I/(a*I**2+b*I+species[2])
    return species, carbon, I_r

###############################
# equivalent to equilibrium in analytical_communities
###############################
    
def equilibrium(species,carbon,I_in,mode = None, approx = False):
    """returns the equilibrium of species under I_in
    
    Parameters:
        species, carbon, I_in: return values of *_par
        mode: None, 'simple, 'partial' or 'full'
            Determines the shape of return value
        approx: boolean
            If False the equilibrium will be calculated with I_out = 0
            
    Returns:
        equi: array
            equilibrium of species at I_in
    """
    try:
        I_inv = I_in.view()  # to not change the shape of I_in
    except AttributeError:
        I_inv = I_in
    # distinguish different cases:
    if isinstance(I_in,(int, float, np.int32, np.float)):
        pass
    elif mode == 'full' or (I_in.ndim==1 and len(I_in)!=species.shape[-1]):
        # equilibrium of each species[...,i] for each entry of I_in[j]
        I_inv.shape = I_in.shape+species[0].ndim*(1,)
    elif mode == 'simple' or (I_in.ndim==1 and len(I_in)==species.shape[-1]):
        # equilibrium of each species[...,i] for the same entry of I_in[i]
        pass
    elif mode=='partial' or (I_in.ndim==2 and I_in.shape[-1]==species.shape[-1]):
        # combination of 'simple' and 'full'. Compute the equilibria of each 
        # species[...,i] for each entry in I_in[:,i]
        I_inv.shape = len(I_in),1,species[0].shape[-1]
    else:
        raise ValueError("""I_in must be a np.array (of dimension 1 or 2) or a 
        scalar, if possible please specify `mode`.""")
    k,l = species[[0,-1]]
    # carbon uptake
    carb_up = lambda I: carbon(species,I,'generating')/(np.expand_dims(k,-1)*I)
    
    # relative light, needed for integration
    rel_I,dx = np.linspace(1e-10,1,21,retstep = True)
    
    #growth rate
    def growth(W):
        I_out = I_inv*np.exp(-k*W) # outcoming light for species        
        # effective lights (linear transformation)
        I_eff = np.expand_dims(I_inv-I_out,-1)*rel_I+np.expand_dims(I_out,-1)
        #integrate carbonuptake, integral computet with linear trans.
        return simps(carb_up(I_eff),dx = dx,axis=-1)*(I_inv-I_out)
    
    # approximated starting density, assumes I_out = 0
    start = growth(np.inf*np.ones(k.shape))/l

    if approx: return start #faster, but might be wrong in certain cases
    # minimal and maximal assumed equilibria, boundaries for secant method
    min_equi = start/2
    max_equi = start*2

    # secant method for finding roots
    for i in range(int(np.log(4000)/np.log(2))):
        av_equi  = (min_equi+max_equi)/2
        test = growth(av_equi)-l*av_equi>0
        min_equi = av_equi*test + min_equi*np.logical_not(test)
        max_equi = av_equi*np.logical_not(test) + max_equi*test
    equi = (min_equi+max_equi)/2 #return equilibria
    return equi
 
###############################
# equivalent to find_balance in analytical_communities
###############################    
    
def find_balance(species, carbon,I_r):
    """finds the incoming light, at which both species have the same I_out*
    
    Parameters:
        species,carbon, I_r: return value of *_par function
            
    Returns:
        I_in: array
        Incoming light at which nonstable coexistence occurs"""
    light_m, light_M = I_r
    I_in = light_m
    k = species[0]
    I_out = k*equilibrium(species, carbon, I_r[0], 'simple')
    dominance = I_out[0]>I_out[1] # which species is better at which conditions
    I_in = 0.5* (light_m+light_M)
    for i in range(15): #iterates until I_in is found
        I_out = k*equilibrium(species, carbon, I_in, 'simple')
        inequal = I_out[0]>I_out[1]
        light_m = light_m+(inequal==dominance)*(I_in-light_m)
        light_M = light_M+(inequal!=dominance)*(I_in-light_M)
        I_in  = 0.5*(light_m+light_M)
    return I_in