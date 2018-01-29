"""
@author: J.W.Spaak

Contains functions to compute the boundary growth rate of species

Species: Method to create species and compute their boundary growth rates

`bound_growth` function to compute the boundary growth rate numerically

`equi_point` function to compute the instable equilibrium

`r_i` numerically computes the growth rate in one period

`dwdt` right hand side of the differential equation
"""

import numpy as np
from scipy.integrate import simps
import numerical_communities as com
from timeit import default_timer as timer

def bound_growth(species, carbon,I_r ,P, num_iterations = 400):
    """computes av(r_i) for species
    
    species, carbon, I_r should be outputs of com.gen_species
    P is the period length
    num_iterstions: Int, number of runs to average over
    
    returns r_i, the boundary growth rate of the two species"""
    
    if I_r.ndim == 1: # species are allowed to have different light ranges
        I_r = np.ones((1,species.shape[-1]))*I_r[:,np.newaxis]
    
    Im, IM = I_r #light range

    acc_rel_I = int(np.sqrt(num_iterations)) # number of simulated lights
    # relative distribution of incoming light in previous time period
    #light in prev period
    rel_I_prev,dx = np.linspace(0,1,acc_rel_I, retstep = True)
    rel_I_prev.shape = -1,1
    # Effective light, linear transformation
    I_prev = (IM-Im)*rel_I_prev+Im
    
    start = timer()
    # equilibrium densitiy of resident in previous period
    dens_prev = com.equilibrium(species, carbon, I_prev, "partial")
    # save the growth rates in the periods
    r_is = np.empty((acc_rel_I, 2,acc_rel_I,species.shape[-1]))
    for i in range(acc_rel_I): #compute the growth rates for each period
        #light in current period
        rel_I_now = np.linspace(0,1,acc_rel_I)[:,np.newaxis]
        I_now = (IM-Im)*rel_I_now+Im
        r_is_save = r_i(dens_prev[i], I_now, species, P, carbon)
        r_is[i] = r_is_save.reshape(2,acc_rel_I,-1, order = 'F')
    # take average via simpson rule, double integral
    av1 = simps(r_is, dx = dx, axis = 2) # integrate over current period
    av2 = simps(av1, dx = dx, axis = 0) # integrate over previous period
    return av2

def r_i(C,E, species,P,carbon):
    """computes r_i for one period, assuming that resident is a equilibrium
    
    it is computed for all incoming lights in E
    
    C: equilibrium density of resident at start of period
    E: incoming light during the period, array
    P: period length
    species, carbon: outputs of com.gen_species
    
    returns r_i for this period"""
    # repeat species, spec_mult[:,:,i<len(E)] \= species[:,:,0]
    spec_mult = species.repeat(len(E), axis = -1)
    C_mult = C.repeat(len(E), axis = -1)
    E_mult = E.reshape(-1,order = 'F') # E_mult = [*E[:,0], E[:,1],E[:,2],...]
    # first axis is for invader/resident, second for the two species
    start_dens = np.array([np.ones(C_mult.shape),C_mult])
    
    dwdt_use = lambda W, t: dwdt(W,t,spec_mult, E_mult, carbon)
    steps = 2*P #number of steps for adams method
    # simulate the growth for the species
    sol = own_ode(dwdt_use, start_dens.reshape(-1), [0,P],steps)
    sol.shape =  steps,2,2,-1 #own_ode only alows 1-dim arrays        
    #divide by P, to compare different period length
    return np.log(sol[-1,0,[1,0]]/1)/P #return in different order, because species are swapped
    
def own_ode(f,y0,t, *args, steps = 10, s = 2):
    """uses adam bashforth method to solve an ODE
    
    Parameters:
        func : callable(y, t0, ...)
            Computes the derivative of y at t0.
        y0 : array
            Initial condition on y (can be a vector).
        t : array
            A sequence of time points for which to solve for y. 
            The initial value point should be the first element of this sequence.
        args : tuple, optional
            Extra arguments to pass to function.
            
    Returns:
        y : array, shape (steps, len(y0))
            Array containing the value of y for each desired time in t,
            with the initial value y0 in the first row."""
    # coefficients for method:
    coefs = [[1], # s = 1
             [-1/2,3/2],  # s = 2
             [5/12, -4/3, 23/12], # s = 3
             [-3/8, 37/24, -59/24, 55/24], # s = 4
             [251/720, -637/360, 109/30, -1387/360, 1901/720]] # s = 5
    coefs = np.array(coefs[s-1]) #choose the coefficients
    ts, h = np.linspace(*t, steps, retstep = True) # timesteps
    # to save the solution and the function values at these points
    sol = np.empty((steps,)+ y0.shape)
    dy = np.empty((steps,)+ y0.shape)
    #set starting conditions
    sol[0] = y0
    dy[0] = f(y0, ts[0], *args)
    # number of points until iteration can start is s
    for i in range(s-1): #use Euler with double precision
        sol_help = sol[i]+h/2*f(sol[i],ts[i], *args)
        sol[i+1] = sol_help+h/2*f(sol_help,ts[i]+h/2)
        dy[i+1] = f(sol[i+1], ts[i+1], *args)
        
    #actual Adams-Method
    for i in range(steps-s):
        sol[i+s] = sol[i+s-1]+h*np.einsum('i,i...->...', coefs, dy[i:i+s])
        dy[i+s] = f(sol[i+s], ts[i+s],*args)
    return sol                
        
def dwdt(W,t,species, I_in,carbon):
    """computes the derivative of the species
    
    Parameters:
        W: array, 
            densities of the species (shape = (2,2,ncom))
        t: not used, allows to be called by odeint
        species, carbon: return values of com.gen_species
        I_in current incoming light
    
    Returns:
        Growth rate of the species
        """
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
    
    from analytical_r_i import mp_approx_r_i
    start = timer()
    species, carbon, I_r = com.gen_species(com.sat_carbon_par, num = 5000)
    print(timer()-start, "generate species")
    P = 20
    ave_r_i = bound_growth(species[:,:,:10], carbon, I_r,P)
    print(timer()-start)
    a,b,c,d = mp_approx_r_i(species[:,:,:10], P, I_r)