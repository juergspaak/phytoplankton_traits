"""
@author: J.W.Spaak

shows that the approximation I_out = 0 is a good one if
one of the species is near its monocultural equilibrium 
for this call show_I_out()
"""
import numpy as np
from scipy.integrate import odeint, simps
from numerical_r_i import dwdt as dwdt
import communities_chesson as ches

def r_i(C,E, species,P,carbon, test = False):
    """computes r_i for the species[:,0] assuming species[:,1] at equilibrium
    
    does this by solving the differential equations
    C should be the density of the resident species, E the incomming light
    P the period length, carbon = [carbon[0],carbon[1]] the carbon uptake 
    functions of both species
    
    returns r_i for this period"""
    # first axis is for invader/resident, second for the two species

    C = ches.equilibrium(species, carbon, np.random.uniform(50,200))
    start_dens = np.array([np.ones(C.shape), C])
    E = np.random.uniform(50,200)
    t = np.linspace(0,P,50)
    sol = odeint(dwdt, start_dens.reshape(-1), t,args=(species, E, carbon))
    sol.shape = len(t),2,2,-1
    rep = np.random.randint(species.shape[-1])
    #### Resident check:
    k,l = species[[0,-1]]
    equi = com.equilibrium(species, E, "simple")
    W_rt = equi-(equi-C)*np.exp(-l*t[:, None, None])
    fig,ax = plt.subplots()
    plt.plot(t,W_rt[:, 0,rep],'.')
    plt.plot(t, sol[:, 1,0,rep])
    plt.plot(t,W_rt[:, 1,rep],'.')
    plt.plot(t, sol[:, 1,1,rep])
    ax.set_ylabel("density of resident", fontsize=14)
    ax.set_xlabel("time", fontsize=14)
    ### invader:
    fig,ax = plt.subplots()
    plt.plot(t,sol[:,0,0,rep])
    plt.plot(t,sol[:,0,1,rep])
    ax.set_ylabel("density of invader", fontsize=14)
    ax.set_xlabel("time", fontsize=14)
    inv = [0,1]
    res = [1,0]

    abso = k[inv]*equi[inv]/(k[res]*equi[res])
    W_it = np.exp((abso[inv]-1)*l[inv]*t[:,None, None])*\
                  (W_rt[:,res]/C[res])**(abso[inv]*l[inv]/l[res])
    plt.plot(t, W_it[:,0, rep], '.')
    plt.plot(t, W_it[:,1, rep], '.')
        
    #divide by P, to compare different period length
    return np.log(sol[1,0,[1,0]]/1)/P #return in different order, because species are swapped
    
spec, carb, I_r = ches.gen_species(ches.sat_carbon_par, num = 5000)
I = np.random.uniform(50,200, spec.shape[-1])
r_i(ches.equilibrium(spec, carb, I), I,spec, 25, carb, True)