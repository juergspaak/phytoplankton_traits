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

from scipy.integrate import simps
from load_pigments import lambs, dlam

def I_in_def(lux, loc = 550, sigma = 0):
    """ returns a incoming light function, a gaussian kernel"""
    if sigma == 0:
        return np.full(lambs.shape, lux)
    else:
        return lux*np.exp(-(lambs-loc)**2/sigma)
        
def spectrum_species(pigments, r_pig, r_spec, n_com, r_pig_spec = 2):
    """
    pigments: list of functions, the pigments
    r_pig: richness of the pigments
    r_spec: richness of species in each community
    n_com: the number of communities to compute
    r_pig_spec: Richness of pigments in each species"""
    n_pig = len(pigments) #number of pigments
    # which pigments are present in which community
    pigs_present_com = np.random.rand(n_pig,n_com).argsort(axis=0)[:r_pig]

    # proportion of pigments for each species
    alpha = np.random.beta(0.1,0.1,(r_pig,r_spec,n_com))

    # each species has at most r_pig_spec pigments
    pigs_present_spec = np.random.rand(r_pig,r_spec,n_com).argsort(axis=0)[:r_pig-r_pig_spec]
    alpha[pigs_present_spec, np.arange(r_spec)[:,None],np.arange(n_com)] = 0
    alpha /= np.sum(alpha, 0)#normalize,each species has same amount of pigments
    k_spec = np.einsum('psc,pcl->lsc', alpha,pigments[pigs_present_com])
    return k_spec, alpha

def multispecies_equi(fitness, k_spec, I_in = I_in_def(40/300),runs = 5000):
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
    pow_fit: float, optional
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
    # starting densities for iteration, shape = (npi, itera)
    equis = np.full(fitness.shape, 1e12) # start of iteration
    equis_fix = np.zeros(equis.shape)

    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = k_spec.copy()
    I_in = I_in[:,np.newaxis]
    unfixed = np.full(fitness.shape[-1], True, dtype = bool)
    n = 20
    i = 0
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    while np.sum(unfixed)/equis.shape[-1]>0.01 and i<runs:          
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = np.einsum('ni,lni->li', equis, abs_points)
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = equis*abs_points #np.einsum('ni,li->nli', equis, abs_points)
        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = all_abs*(I_in/tot_abs*(1-np.exp(-tot_abs)))[:,np.newaxis]
        # fit*int(y_simps)
        equis = fitness*simps(y_simps, dx = dlam, axis = 0)
        # remove rare species
        equis[equis<1] = 0
        if i % n==n-2:
            # to check stability of equilibrium in next run
            equis_old = equis.copy()
        if i % n==n-1:
            
            stable = np.logical_or(equis == 0, #no change or already 0
                                   np.abs((equis-equis_old)/equis)<1e-3)
            cond = np.logical_not(np.prod(stable, 0)) #at least one is unstable
            equis_fix[:,unfixed] = equis #copy the values
            # prepare for next runs
            unfixed[unfixed] = cond
            equis = equis[:,cond]
            abs_points = abs_points[...,cond]
            fitness = fitness[:,cond]
        i+=1
    #return only communities that found equilibrium
    return equis_fix, unfixed