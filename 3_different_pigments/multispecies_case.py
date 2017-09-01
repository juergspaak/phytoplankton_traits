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
from load_pigments import real,rand

def I_in_def(lux, loc = 550, sigma = 0):
    """ returns a incoming light function, a gaussian kernel"""
    if sigma == 0:
        return lambda lam: np.full(lam.shape, lux)
    else:
        return lambda lam: lux*np.exp(-(lam-loc)**2/sigma)
        
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
    # convert functions to arrays, pigments might be inefficient
    lam = np.linspace(400,700,101)
    pig_lam = np.array([pig(lam) for pig in pigments])
    # proportion of pigments for each species
    alpha = np.random.beta(0.1,0.1,(r_pig,r_spec,n_com))

    # each species has at most r_pig_spec pigments
    pigs_present_spec = np.random.rand(r_pig,r_spec,n_com).argsort(axis=0)[:r_pig-r_pig_spec]
    alpha[pigs_present_spec, np.arange(r_spec)[:,None],np.arange(n_com)] = 0
    alpha /= np.sum(alpha, 0)#normalize,each species has same amount of pigments
    k_spec = np.einsum('psc,pcl->lsc', alpha,pig_lam[pigs_present_com])
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

    lam, dx = np.linspace(400,700,101, retstep = True) #points for integration
    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = k_spec.copy()
    I_in = I_in(lam)[:,np.newaxis]
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
        equis = fitness*simps(y_simps, dx = dx, axis = 0)
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
    spec_num = np.sum(equis>0,0)
    fig = plt.figure()
    plt.plot(np.linspace(0,100, len(spec_num)),sorted(spec_num))
    plt.ylabel("number coexisting species")
    plt.xlabel("percent")
    plt.show()
    return fig
    
def plot_percentile_curves():
    """plots the percentile curves (5,25,50,75 and 95), of the number of coexisting
    species in dependence of the number of pigments"""
    equis = []
    unfixeds = []
    pigs_richness = np.arange(2,21,2)
    for i in pigs_richness:
        n_com = 500 # number of communities
        r_spec = 2*i # richness of species, regional richness
        r_pig = i #richness of pigments in community
        r_pig_spec = min(i,5) #richness of pigments in each species
        fac = 2 #factor by which fitness can differ
        phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com)) #photosynthetic efficiency
        l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com)) # specific loss rate
        fitness = phi/l # fitness
        k_spec,alpha = spectrum_species(rand, r_pig, r_spec, n_com, r_pig_spec) #spectrum of the species
        equi, unfixed = multispecies_equi(fitness, k_spec)
        equis.append(equi[:,np.logical_not(unfixed)]) #equilibrium density
        unfixeds.append(unfixed)
        plt_ncoex(equis[-1])
        print(i)
    spec_nums = [np.sum(equi>0,0) for equi in equis]    
    percents = np.array([[int(np.percentile(spec_num,per)) for per in [5,25,50,75,95]] for
                 spec_num in spec_nums])
    fig, ax = plt.subplots()
    leg = plt.plot(pigs_richness,percents, '.')
    ax.set_ylabel("number of coexisting species")
    ax.set_xlabel("number of pigments in the community")
    ax.legend(leg, ["5%","25%","50%","75%","95%"], loc = "upper left")
    ax.set_ybound(np.amin(percents)-0.1, np.amax(percents)+0.1)
    plt.figure()
    plt.plot(k_spec[:,:,0]) #plot a representative of the spectrum
    return equis, unfixeds, percents