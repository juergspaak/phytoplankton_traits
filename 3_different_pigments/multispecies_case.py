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
        
def spectrum_species(pigments, r_pig, r_spec, n_com):
    """
    pigments: list of functions, the pigments
    r_pig: richness of the pigments
    r_spec: richness of species in each community
    n_com: the number of communities to compute"""
    n_pig = len(pigments) #number of pigments
    # which pigments are present in which community
    pigs_present = np.random.rand(n_pig,n_com).argsort(axis=0)[:r_pig]
    # convert functions to arrays, pigments might be inefficient
    lam = np.linspace(400,700,101)
    pig_lam = np.array([pig(lam) for pig in pigments])
    # proportion of pigments for each species
    alpha = np.random.beta(0.1,0.1,(r_pig,r_spec,n_com))
    alpha /= np.sum(alpha, 0)#normalize,each species has same amount of pigments
    k_spec = np.einsum('psc,pcl->lsc', alpha,pig_lam[pigs_present])
    return k_spec

def multispecies_equi(fitness, k_spec, I_in = I_in_def(40/300),runs = 500):
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
    equis = np.full(fitness.shape, 1e10) # start of iteration
    equis_fix = np.zeros(equis.shape)

    lam, dx = np.linspace(400,700,101, retstep = True) #points for integration
    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = k_spec.copy()
    I_in = I_in(lam)[:,np.newaxis]
    unfixed = np.full(fitness.shape[-1], True, dtype = bool)
    n = 20
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    for i in range(runs):          
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
            print(i, np.sum(unfixed))
    #return only communities that found equilibrium
    return equis_fix#[:,np.logical_not(unfixed)]
    
def stomp_par(num = 1000,richness = 3, fac = 4):
    """ returns random parameters for the model
    
    the generated parameters are ensured to survive
    
    the carbon uptake function depends on the light spectrum
    
    returns species, which contain the different parameters of the species
    species_x = species[:,x]
    carbon = [carbon[0], carbon[1]] the carbon uptake functions
    I_r = [Im, IM] the minimum and maximum incident light
    richness is the species richness, number of species per community"""
    phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(richness,num))
    l = 0.014*np.random.uniform(1/fac, 1*fac,(richness,num))
    pigments = random_pigments(num, richness-1)
    # raise by
    pig_ratio = np.random.beta(0.1,0.1, (richness-1,richness, num))
    k_spec = lambda lam: np.einsum('psn,lpn->lsn', pig_ratio, pigments(lam))
    return phi/l, k_spec
    
    
def random_pigments(num, richness, n_peak_max = 3):
    """randomly generates `n` absorption spectra
    
    n: number of absorption pigments to create
    
    Returns: pigs, a list of functions. Each function in pigs (`pigs[i]`) is
        the absorption spectrum of this pigment.
    `pig[i](lam)` = sum_peak(gamma*e^(-(lam-peak)**2/sigma)"""
    shape = (n_peak_max, richness, num)
    
    # number of peaks for each pigment:
    n_peaks = np.random.binomial(1,0.5,shape)
    n_peaks[0] = 1 #each phytoplankton has at least 1 peak
    # location of peaks
    l_p = np.random.uniform(400,700,shape)
    # shape of peak
    sigma_p = np.random.uniform(100,900, shape)
    # magnitude of peak, multiply by n_peaks, as not all peaks might be present
    gamma_p = np.random.uniform(0,1, shape)*n_peaks
    
    # absorption of pigments
    pigments = lambda lam: np.sum(gamma_p*np.exp(-(lam.reshape(-1,1,1,1)
                                -l_p)**2/sigma_p),axis = 1)
    
    # uniformize all pigments (have same integral on [400,700])
    lams,dx = np.linspace(400,700,301, retstep = True)
    absorption = pigments(lams)
    # total absorption
    energy = simps(absorption,dx = dx, axis = 0)
    gamma_p /= energy #divide by total absorption
    def pigments(lam):
        b = np.sum(gamma_p*np.exp(-(lam.reshape(-1,1,1,1)-l_p)**2/sigma_p),axis = 1)
        a = np.amax([1e-14*np.ones(b.shape),b],axis = 0)
        return a/1e9
    pigments.parameters = {'gamma_p': gamma_p, 'l_p': l_p, 'sigma_p': sigma_p}
    return pigments
    
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
    spec_num = np.sum(equi>0,0)
    fig = plt.figure()
    plt.plot(np.linspace(0,100, len(spec_num)),sorted(spec_num))
    plt.ylabel("number coexisting species")
    plt.xlabel("percent")
    return fig

n_com = 1000
r_spec = 4
r_pig = 4
fac = 2
phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))
l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com))    
fitness = phi/l
k_spec = spectrum_species(rand, r_pig, r_spec, n_com)
equi = multispecies_equi(fitness, k_spec)
plt_ncoex(equi)
plt.figure()
plt.plot(k_spec[:,:,0])