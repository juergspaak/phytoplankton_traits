# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:27:36 2017

@author: spaakjue
"""
import numpy as np
import stomp_communities as com

def r_i(C,I_in, species,g_spec,k_spec, P):
    """computes r_i for one period, assuming that resident is a equilibrium
    
    it is computed for all incoming lights in E
    
    C: equilibrium density of resident at start of period
    E: incoming light during the period, array
    P: period length
    species, carbon: outputs of com.gen_species
    
    returns r_i for this period"""
    # first axis is for invader/resident, second for the two species
    
    steps = int(P/10) #number of steps for adams method
    # simulate the growth for the species
    sol = own_ode(g_spec, C, [0,P],args = (species, k_spec, I_in), steps = steps)
    sol.shape =  steps,*C.shape #own_ode only alows 1-dim arrays        
    #divide by P, to compare different period length
    return sol

def own_ode(f,y0,t,args = (), steps = 10, s = 2 ):
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
        sol[i+1] = sol_help+h/2*f(sol_help,ts[i]+h/2, *args)
        dy[i+1] = f(sol[i+1], ts[i+1], *args)
        
    #actual Adams-Method
    for i in range(steps-s):
        sol[i+s] = sol[i+s-1]+h*np.einsum('i,i...->...', coefs, dy[i:i+s])
        dy[i+s] = f(sol[i+s], ts[i+s], *args)
    return sol
    
def I_out(density, k_spec, I_in):
    lambs = np.linspace(400,700,151)
    absor = density*k_spec(lambs)
    tot_abs = np.sum(absor, axis = 1) # \sum_i  k_i(lam)*N_i
    I_in = I_in(lambs).reshape(-1,1)
    # integrate_400^700 rel_absor*I_in(lambda)*(1-e^(-absorption)) dlambda
    energy = I_in*(1-np.exp(-tot_abs))
    print(energy.shape)
    plt.plot(np.linspace(400,700,151),energy[:,0])
    plt.figure()
    plt.plot(np.linspace(400,700,151),k_spec(np.linspace(400,700,151))[:,:,0])
    
def pigment_fraction(N, pigment,k_spec, I_in = com.I_in_def(40/300)):
    lams,dx = np.linspace(400,700,51, retstep = True)
    abs_points = k_spec(lams) 
    tot_abs = np.einsum('ni,lni->li', N, abs_points)
    p_lam = pigment(lams)
    I_lam = I_in(lams)[:,np.newaxis]
    y_simps = p_lam*(I_lam/tot_abs*(1-np.exp(-tot_abs)))[:,np.newaxis]
    return simps(y_simps, dx = dx, axis = 0)
    
    
def pigments_equal():    
    """this code snippet proves, that independent of the incoming light, the 
    energy produced by one pigment is always the same at equilibrium"""
    species, k_spec, g_spec = com.stomp_par(richness = 3, num = 1000)
    lux = np.random.uniform(10/300, 300/300,2)
    sigma = np.random.uniform(5000,50000,2)
    loc = np.random.uniform(400,700,2)
    I_in0 = com.I_in_def(lux[0], loc[0], sigma[0])
    I_in2 = com.I_in_def(lux[1], loc[1], sigma[1])
    equi0  = com.equilibrium(k_spec, species, I_in0)
    equi2  = com.equilibrium(k_spec, species, I_in2)
    surv0 = equi0>0
    surv2 = equi2>0
    pig_f0 = pigment_fraction(equi0, k_spec.pigments, k_spec,I_in0)
    pig_f2 = pigment_fraction(equi2, k_spec.pigments, k_spec,I_in2)
    
    interest = np.logical_and(np.sum(surv0,0)==2, np.sum(surv2,0)==2)
    print(interest.sum())
    print(pig_f0[:,interest])
    print(pig_f2[:,interest])
    print(np.amax(np.abs(pig_f0[:,interest]-pig_f2[:,interest])/pig_f0[:,interest]))
    lambs = np.linspace(400,700,151)
    plt.plot(lambs, I_in0(lambs))
    plt.plot(lambs, I_in2(lambs))
    
species, k_spec, g_spec = com.stomp_par(richness = 3, num = 1000)
lux = np.random.uniform(10/300, 300/300,2)
sigma = np.random.uniform(5000,50000,2)
loc = np.random.uniform(400,700,2)
I_in0 = com.I_in_def(lux[0], loc[0], sigma[0])   
equi0  = com.equilibrium(k_spec, species, I_in0)
surv0 = equi0>0
interest = np.sum(surv0,0)==2
fit = (species[0]/species[1])[:,interest]
inv = np.argmax(fit,0)
inv2 = np.argmin(fit,0)
surv = surv0[:,interest]

print([surv[inv[i],i] for i in range(len(inv))])
new = ~np.array([surv[inv[i],i] for i in range(len(inv))])
print(fit[:,new])
print(np.sum(new), new.size)
"""    
I_in = lambda lam: 25/300*np.ones(lam.shape)
species, k_spec, g_spec = com.stomp_par(richness = 3, num = 1000)
sol = r_i(1000*np.ones(species[0].shape), I_in,
          species, g_spec, k_spec,10000)
sol2 = com.equilibrium(k_spec, species, I_in).swapaxes(0,1)
print(np.nanargmin(sol[-1]))"""
"""
I_in = lambda lam: 100/300*np.ones(lam.shape)
sol2 = r_i(1000*np.ones(species[0].shape), I_in,
          species, g_spec, k_spec,10000)
surv = sol[-1]>1e5
surv2 = sol2[-1]>1e5
diff = np.logical_xor(surv, surv2)
interest = np.sum(diff, axis = 0)==2
richness_const = np.sum(surv, axis = 0)==np.sum(surv2, axis = 0)
wuhu = np.logical_and(interest, richness_const)
print(np.sum(wuhu))
print(sol[-1,:,wuhu])
print(sol2[-1,:,wuhu])
I_out(sol[-1,:,:], k_spec, I_in)"""
