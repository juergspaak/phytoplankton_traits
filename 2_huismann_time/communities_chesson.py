"""
@author: J.W.Spaak

Generates random species

`gen_species` returns two random species

`sat_carbon_par`: rand. generate two species with saturating carbon uptake
    functions
`photoinhibition_par`: rand. generate two species with carbon uptake
    functions that suffer from photoinhibition
    
`equilibrium` computes the equilibrium density of the species


"""
import numpy as np
from numpy.random import uniform as uni
from scipy.integrate import simps

def gen_species(parameter_generator, num = 1000):
    """returns two species, for which dominance depends on I_in
    
    parameter_generator: sat_carbon_par or photoinhibition_par
    
    returns:
        same values as parameter_generator"""
    species, carb,I_r = parameter_generator(num)
    def carbon(spec, I, mode = 'full'):
        if isinstance(I,(int, float)) or mode == 'simple':
            pass
        elif mode == 'full':
            I_shape = I.ndim*(1,)
            spec = [par.reshape(par.shape+I_shape) for par in spec]
            I = I.reshape(spec[0].ndim*(1,)+I.shape)
        elif mode == 'partial':
            I_shape = (I.ndim-1)*(1,)
            I = I.reshape((1,)+I.shape[::-1])
            spec = [par.reshape(par.shape+I_shape) for par in spec]
        elif mode == 'generating':
            spec = np.expand_dims(spec,-1)
        elif mode == 'special':
            spec = spec.reshape(spec.shape+(1,1)).swapaxes(-3,-2)
            I = np.rollaxis(I,1,2)
            I = I.reshape((1,)+I.shape)
        return carb(spec, I)
    Im, IM = I_r
    k = species[0]

    # compute absorption of both species (equivalent to I_out)
    equis = equilibrium(species, carbon, I_r, 'full')
    k[0,np.newaxis]*equis[0]
    dominance = k[0,np.newaxis]*equis[0]>k[1,np.newaxis]*equis[1] #dominance of species
    # balanced if dominance changes
    balanced = np.logical_xor(dominance[0], dominance[1])
    # species must have positive aboundancies
    survive = np.amin(equis, axis = 1)>0
    #species that are balanced and survive
    good = np.logical_and(balanced, np.logical_and(survive[0], survive[1]))
    return species[:,:,good], carbon, I_r


                  
def equilibrium(spec,carbon,I_in,mode = 'full', approx = False):
    """returns the equilibrium of species under light_0
    species and light_0 can either be an array or a number
    light_0 needs the same dimention as species"""
    # distinguish different cases:
    if isinstance(I_in,(int, float, np.int32, np.float)):
        rel_I_shape = (1,1,-1)
    elif mode == 'simple' or (I_in.ndim==1 and len(I_in)==spec.shape[-1]):
        I_in = I_in*np.ones(spec[0].shape)
        rel_I_shape = (1,1,-1)
    elif mode=='partial' or (I_in.ndim==2 and I_in.shape[-1]==spec.shape[-1]):
        I_shape = I_in.ndim*(1,)
        I_in = I_in.reshape((spec[0].ndim-1)*(1,)+I_in.shape).swapaxes(1,2)
        spec = np.array([par.reshape(par.shape+(1,)) for par in spec])
        rel_I_shape = (1,1,1,-1)
    elif mode == 'full' or (I_in.ndim==1 and len(I_in)!=spec.shape[-1]):
        I_shape = I_in.ndim*(1,)
        I_in = I_in.reshape(spec[0].ndim*(1,)+I_in.shape)
        spec = np.array([par.reshape(par.shape+I_shape) for par in spec])
        rel_I_shape = (1,1,1,-1)
    
    
    k,l = spec[[0,-1]]
    # carbon uptake
    carb_up = lambda I: carbon(spec,I,'generating')/(np.expand_dims(k,-1)*I)
    
    # relative light, needed for integration
    rel_I,dx = np.linspace(1e-10,1,21,retstep = True)
    rel_I = rel_I.reshape(rel_I_shape)
    
    #growth rate
    def growth(W):
        I_out = I_in*np.exp(-k*W) # outcoming light for species        
        # effective lights (linear transformation)
        I_eff = np.expand_dims(I_in-I_out,-1)*rel_I+np.expand_dims(I_out,-1)
        #integrate carbonuptake, integral computet with linear trans.
        return simps(carb_up(I_eff),dx = dx,axis=-1)*(I_in-I_out)
    
    # approximated starting density, assumes I_out = 0
    start = growth(np.inf*np.ones(k.shape))/l

    if approx: return start #faster, but might be wrong in certain cases
    # minimal and maximal assumed equilibria, boundaries for secant method
    min_equi = start/2
    max_equi = start*2

    # secant method for finding roots
    for i in range(int(np.log(4*1000)/np.log(2))):
        av_equi  = (min_equi+max_equi)/2
        test = growth(av_equi)-l*av_equi>0
        min_equi = av_equi*test + min_equi*np.logical_not(test)
        max_equi = av_equi*np.logical_not(test) + max_equi*test
    equi = (min_equi+max_equi)/2 #return equilibria
    if equi.ndim == 3:
        return equi.swapaxes(1,2)
    return equi

def sat_carbon_par(num = 1000, factor=4,Im = 50.0, IM = 200.0):
    """ returns random parameters for the model
    the generated parameters are ensured to survive
    
    the carbon uptake function is saturating (no photoinhibition)
    
    returns species, which contain the different parameters of the species
    species_x = species[:,x]
    carbon = [carbon[0], carbon[1]] the carbon uptake functions
    I_r = [Im, IM] the minimum and maximum incident light"""
    fac = np.sqrt(factor)
    k = uni(0.004/fac, 0.004*fac,(2,num)) # absorption coefficient [m^2/g]
    H = uni(100/fac, 100*fac,(2,num)) # halfsaturation for carbon uptake [J/(m^2 s)]
    p_max = uni(9/fac, 9*fac,(2,num)) # maximal absorbtion of carbon [s^-1]
    l = uni(0.5/fac, 0.5*fac,(2,num))  # carbon loss [s^-1]
    species = np.array([k,H,p_max,l])
    def carbon(spec, I):
        return spec[2]*I/(spec[1]+I)
    return species, carbon, np.array([Im, IM])
    
def photoinhibition_par(num = 1000, factor=4, Im = 100., IM = 500.):
    """ returns random parameters for the model
    
    the carbon uptake function suffers from photoinhibition
    
    returns species, which contain the different parameters of the species
    species_x = species[:,x]
    carbon = [carbon[0], carbon[1]] the carbon uptake functions
    I_r = [Im, IM] the minimum and maximum incident light"""
    fac = np.sqrt(factor)
    k = uni(0.002/fac, fac*0.002,(2,num))
    p_max = uni(1/fac, fac*1,(2,num))
    I_k = uni(40/fac, fac*40,(2,num))
    I_opt = uni(100/fac, fac*100,(2,num))
    l = uni(0.5/fac, 0.5*fac,(2,num))  # carbon loss
    species = np.array([k,p_max,I_k, I_opt, l])
    def carbon(spec, I):
        a = spec[2]/spec[3]**2
        b = 1-2*spec[2]/spec[3]
        return I*spec[0]/(a*I**2+b*I+spec[2])
    return species, carbon, np.array([Im, IM])

def find_balance(species, carbon,I_r):
    """finds the incoming light, at which both species have the same I_out*
    
    Parameters:
        species: return value of *_par function such that one species dominates
            at I_in = 200 and the other at I_in = 50
            
    Returns:
        I_in: float
        Incoming light at which nonstable coexistence occurs"""
    light_m, light_M = I_r
    I_in = light_m
    k = species[0]
    I_out = k*equilibrium(species, carbon, I_r[0])
    dominance = I_out[0]>I_out[1] # which species is better at which conditions
    I_in = 0.5* (light_m+light_M)
    for i in range(15): #iterates until I_in is found
        I_out = k*equilibrium(species, carbon, I_in, 'simple')
        inequal = I_out[0]>I_out[1]
        light_m = light_m+(inequal==dominance)*(I_in-light_m)
        light_M = light_M+(inequal!=dominance)*(I_in-light_M)
        I_in  = 0.5*(light_m+light_M)
    return I_in
    
spec, carb, I_r = gen_species(sat_carbon_par, num = 500)