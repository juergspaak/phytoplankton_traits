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
from scipy.integrate import simps,odeint
import numerical_communities as com

def bound_growth(species, carbon,I_r ,P):
    """computes av(r_i) for species
    
    species, carbon, I_r should be outputs of gen_species
    I_r should be the range in which I_in fluctuates
    
    returns ave_C, ave_E, r_i = ave_E-ave_C
    
    ave_E is assumed to be very small"""
    if I_r.ndim == 1:
        I_r = np.ones((1,species.shape[-1]))*I_r[:,np.newaxis]
    
    Im, IM = I_r #light range
    # density in dependence of incoming light
    acc_rel_I = 1000 # number used for simpson integration
    # uniform distribution in 0,1, used for integration
    rel_I_prev = np.sort(np.random.random((acc_rel_I,1)))
    rel_I_now = np.random.random((acc_rel_I,1)) 
    # Effective light used for integration
    I_prev = (IM-Im)*rel_I_prev+Im
    I_now = (IM-Im)*rel_I_now+Im
    start = timer()
    dens_prev = com.equilibrium(species, carbon, I_prev, "partial")
    mid = timer()
    r_is = np.empty((acc_rel_I,2,species.shape[-1]))
    mido = timer()
    for i in range(acc_rel_I):
        r_is[i] = r_i(dens_prev[:,i], I_now[i], species, P, carbon)
    end = timer()
    print(end-mido, mid-start)
    return np.average(r_is, axis = 0)

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
    dwdt_use = lambda W, t: dwdt(W,t,species, E, carbon)
    steps = 2*P
    sol = own_ode(dwdt_use, start_dens.reshape(-1), [0,P],steps)
    sol.shape =  steps,2,2,-1 #odeint only alows 1-dim arrays        
    #divide by P, to compare different period length
    return np.log(sol[-1,0,[1,0]]/1)/P #return in different order, because species are swapped
    
    
def own_ode(f,y0,t, steps = 10, s = 2):
    # coefficients for method:
    coefs = [[1],
             [-1/2,3/2],
             [5/12, -4/3, 23/12],
             [-3/8, 37/24, -59/24, 55/24],
             [251/720, -637/360, 109/30, -1387/360, 1901/720]]
    coefs = np.array(coefs[s-1])
    ts, h = np.linspace(*t, steps, retstep = True)
    sol = np.empty((steps,)+ y0.shape)
    dy = np.empty((steps,)+ y0.shape)
    sol[0] = y0
    dy[0] = f(y0, ts[0])
    for i in range(s-1):
        sol_help = sol[i]+h/2*f(sol[i],ts[i])
        sol[i+1] = sol_help+h/2*f(sol_help,ts[i]+h/2)
        dy[i+1] = f(sol[i+1], ts[i+1])
    for i in range(steps-s):
        sol[i+s] = sol[i+s-1]+h*np.einsum('i,i...->...', coefs, dy[i:i+s])
        dy[i+s] = f(sol[i+s], ts[i+s])
    return sol                
        
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
    spec, carbon, I_r = com.gen_species(com.sat_carbon_par, num = 5000)
    a,b,c = bound_growth(spec, carbon, I_r,50)    
    
if False: # check r_i is doing a good job
    I = np.random.uniform(50,200, spec.shape[-1])
    r_i(com.equilibrium(spec, carbon, I), I,spec, 10, carbon, True)
    
#spec, carbon, I_r = com.gen_species(com.photoinhibition_par, num = 500000)
plit(*save, save = "different carbon_uptake")    
#save = bound_growth(spec[:,:,:1000], carbon,I_r ,10)
#print(np.abs(save[0]-r_i3[1,:10])/np.max(np.abs([save[0],r_i3[1,:10]]), axis = 0))
#print(save[0])