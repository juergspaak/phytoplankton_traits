"""
@author: J. W. Spaak, jurg.spaak@unamur.be

This file is to find the amount of different pigments that exit in nature
To understand the mathematical equations look at 
"../../4_analysis/3.Finite pigments/3. Finite pigments.docx", 
Assumptions: z_m = 1, Sum(\alpha_i)=1

`multispecies_equis` computes the equilibrium density of multiple species in
    one community and returns these equilibria

`plt_ncoex`plots how many species coexist in how many iterations.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
    
def multispecies_equi(pigs, itera = int(1e4),runs = 100, av_fit =1.4e8,
                        pow_fit = 2, per_fix = True, sing_pig = False):
    """Compute the equilibrium density for several species with its pigments
    
    Computes `itera` randomly selected communities, each community contains
    at most len(`pigs`) different species. Returns equilibrium densities
    for each community.
    
    Parameters
    ----------
    pigs : list of functions
        Each element `pig` of pigs must be the absorption spectrum of that 
        pigment. Each `pig` must be a function that returns a float
    itera : int, optional
        number of generated communities
    runs : int, optional
        number of iterations to find equilibrium
    av_fit: float, optional
        Average fitness of all species
    pow_fix: float, optional
        Fitness of each species ill be in [1/pow_fit, pow_fit]*av_fit
    per_fix: Bool, optional
        Percent of fixed species is printed if True
    sing_pig: Bool, optional
        Determines if species have only one pigment. If False, species 
        absorption spectrum will be a sum of different pigments        
        
    Returns
    -------
    equis:
        Equilibrium densitiy of all species, that reached equilibrium      
    """
    # total number of species
    npi = len(pigs)
    # asign a fitness to all species, shape = (npi, itera)
    fitness = 2**np.random.uniform(-1,1,[npi, itera])*1.4e8
    # not all species are present in all communities
    pres = np.random.binomial(1, 0.8, [npi, itera])
    fitness = pres*fitness
    # starting densities for iteration, shape = (npi, itera)
    equis = 1e20*np.ones([npi, itera]) # start of iteration
    
    lam, dx = np.linspace(400,700,101, retstep  = True) #points for integration
    # k_j(lam), shape = (len(fits), len(lam))
    abs_points = np.array([pig(lam) for pig in pigs])
    str_abs = 'nl'
    
    if not sing_pig:
        # how much of which pigment is in which species, shape = (npi, npi, itera)
        alpha = np.random.uniform(size = [npi, npi, itera])
        #sum of all coefficients must be 1
        alpha = np.einsum('nki,ni->nki',alpha, 1/np.sum(alpha,1))
        # sum_j(a_ij*k_j(lam)), shape = (npi, len(lam), itera)
        abs_points = np.einsum('nki, kl->nli', alpha, abs_points)
        str_abs = 'nli'
 
    for i in range(runs):
        if i == runs-1:
            #save previous values in final run to see whether equilibrium has 
            #been reached
            equis_old = equis.copy()
        if i%(int(runs/10))==0: #prograss report
            print(100*i/runs, "percent done")
            
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = np.einsum('ni,'+str_abs+'->li', equis, abs_points)
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = np.einsum('ni,'+str_abs+'->nli', equis, abs_points)
        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = np.einsum('nli,li->nli', 
                            all_abs, (1-np.exp(-tot_abs))/tot_abs)
        # fit*int(y_simps)
        equis = fitness*simps(y_simps, dx = dx, axis =1)
        # remove rare species
        equis[equis<1] = 0
    # exclude the species that have not yet found equilibrium, avoid nan
    stable = np.logical_or(equis == 0,(equis-equis_old)/equis_old<0.0001)
    if per_fix:
        print("percent of species that reached equilibrium:",
              np.sum(stable)/stable.size)
    equis = stable*equis
    
    #group the species that belong into one community
    return np.array([equis[i].reshape(-1) for i in range(len(equis))]).T
    
def plt_ncoex(equis):
    """plots the amount of different species in a percentile curve
    
    equis: 2-dim array
        equis[i] should be the equilibrium densities of the species in
        iteration i
    
    Returns: None
    
    Plots: A percentile plot of the number of species that coexist in each
        iteration
        
    Example: 
        import load_pigments as pigments
        plt_ncoex(multispecies_equi(pigments.real))"""
    spec_num = [np.count_nonzero(i) for i in equis]
    fig = plt.figure()
    plt.plot(np.linspace(0,100, len(spec_num)),sorted(spec_num))
    plt.ylabel("number coexisting species")
    plt.xlabel("percent")
    return fig