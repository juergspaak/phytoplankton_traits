"""
@author: J.W.Spaak

Debug functions
Contain functions to check, that the other functions work correctly
"""
import numpy as np
import matplotlib.pyplot as plt

# import all files for Debugfunctions
import numerical_communities as numcom
import numerical_r_i as numri
import analytical_communities as anacom
import analytical_r_i as anari

# create the species
# numerical species with saturating carbon uptake
spec_sat, carb_sat, I_r_sat = numcom.gen_species(numcom.sat_carbon_par, 10000)
# numerical species with photoinhibition
spec_inh, carb_inh,I_r_inh=numcom.gen_species(numcom.photoinhibition_par,1000)
# analytical species with saturating carbon uptake
spec_ana = anacom.gen_species(1000)
I_r_ana = I_r_sat.copy()

###############################################################################
# Check that similar functions give similar results

def same_equilibria():
    """ to check that the equilibrium function of both files are the same"""
    I_in = np.random.uniform(*I_r_sat, spec_sat.shape[-1])
    equi_num = numcom.equilibrium(spec_sat, carb_sat, I_in, 'simple')
    equi_ana = anacom.equilibrium(spec_sat, I_in, 'simple')
    rel_diff = np.sort((np.abs(equi_num-equi_ana)/equi_ana).ravel())
    fig,ax = plt.subplots()
    ax.plot(np.linspace(0,100,rel_diff.size),rel_diff)
    ax.semilogy()
    ax.set_ylabel("relative difference in equilibria")
    ax.set_xlabel("percentiles")
    
def same_find_balance():
    """ to check that the equilibrium function of both files are the same"""
    bal_num = numcom.find_balance(spec_sat, carb_sat, I_r_sat)
    bal_ana = anacom.find_balance(spec_sat, I_r_sat)
    rel_diff = np.sort((np.abs(bal_num-bal_ana)/bal_ana).ravel())
    fig,ax = plt.subplots()
    ax.plot(np.linspace(0,100,rel_diff.size),rel_diff)
    ax.semilogy()
    ax.set_ylabel("relative difference in balance")
    ax.set_xlabel("percentiles")

###############################################################################
# Show that the assumption I_out = 0 is a good one

def plot_carbon_uptake():
    """show the different carbon uptake functions"""
    I_r = [0,1000]
    data_sat = carb_sat(spec_sat, np.linspace(*I_r, 100), 'generating')
    data_inh = carb_inh(spec_inh, np.linspace(*I_r, 100), 'generating')
    fig,ax = plt.subplots()
    ax.set_ylabel(r'$p(I)$')
    ax.set_xlabel(r'$I$')
    for k in range(5):
        i = np.random.randint(2)
        plt.plot(np.linspace(*I_r, 100),data_sat[i,k], '--')
        plt.plot(np.linspace(*I_r, 100),data_inh[i,k], ',')

def I_out_zero_approx():
    """solves growth rates with assuming I_out =0 and compares to exact"""
    # first axis is for invader/resident, second for the two species
    from scipy.integrate import odeint
    from numerical_r_i import dwdt as dwdt
    
    P = 25 #period length
    C = numcom.equilibrium(spec_sat, carb_sat, np.random.uniform(50,200))
    start_dens = np.array([np.ones(C.shape), C])
    E = np.random.uniform(10,200)
    t = np.linspace(0,P,50)
    sol = odeint(dwdt, start_dens.reshape(-1), t,args=(spec_sat, E, carb_sat))
    sol.shape = len(t),2,2,-1
    rep = np.random.randint(spec_sat.shape[-1])
    #### Resident check:
    k,l = spec_sat[[0,-1]]
    equi = anacom.equilibrium(spec_sat, E, "simple")
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

    
"""functions to include:
    
    compare different r_i functions
    compare r_i analytical with r_i numerical with sat_carbon
    Adapt photo_inh pars to fit -sat pars to compare solutions"""