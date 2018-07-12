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
        return np.full(lambs.shape, lux, dtype = "float")/300
    else:
        prelim = np.exp(-(lambs-loc)**2/(2*sigma**2))
        return lux*prelim/simps(prelim, dx = dlam)
        
def spectrum_species(pigments, r_pig, r_spec, n_com, r_pig_spec = 2):
    """
    pigments: absorptions at `lambs`, the pigments in all communities
    r_pig: richness of the pigments in each community
    r_spec: richness of species in each community (regional richness)
    n_com: the number of communities to generate
    r_pig_spec: Richness of pigments in each species"""
    n_pig = len(pigments) #number of pigments
    # which pigments are present in which community
    pigs_present_com = np.random.rand(n_pig,n_com).argsort(axis=0)[:r_pig]
    # each species has at most r_pig_spec pigments
    pigs_present_spec = np.random.rand(r_spec, r_pig,n_com).argsort(axis=1)\
                    [:,:r_pig_spec]
    # which species contains how much of which pigment
    alpha = np.full((n_pig, r_spec, n_com),0, dtype = "float")
    
    data = pigs_present_com[pigs_present_spec, np.arange(n_com)]
    new = np.repeat(np.arange(r_spec),data.size//r_spec).reshape(data.shape)
    # proportion of pigments for each species
    alpha[data, new, np.arange(n_com)] = np.random.beta(0.1,0.1,(data.shape))
    # check alpha for correctness
    check = alpha>0
    # check each species has r_pig_spec pigments
    if not (np.sum(check, axis = 0)==r_pig_spec).all():
        raise
    # check each communitiy has at most r_pig pigments
    if not ((np.sum(check, axis =1)>0).sum(axis = 0)<r_pig+1).all():
        raise
    # normalize,each species has same amount of expected pigments
    alpha /= r_pig_spec/2
    # absorption spectrum of the species
    k_spec = np.einsum('psc,pl->lsc', alpha,pigments)
    return k_spec, alpha
    
def multispecies_equi(fitness, k_spec, I_in = I_in_def(40),runs = 5000, 
                      k_BG = 0):
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
    equis = np.full(fitness.shape, 1e7) # start of iteration
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
    
if __name__ == "__main__":
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    pigments = real[[0,5,6]] # chlo_a, phycoerythrin, phycocyanin
    plt.plot(lambs,pigments.T)
    plt.title("Pigments used")
    n_com = 2
    k_spec, alpha = spectrum_species(pigments, 3, 2, n_com)
    plt.figure()
    plt.title("Absorption of species")
    plt.plot(lambs, k_spec[...,0])
    phi = 2*1e8*np.random.uniform(0.9,1.1,(2,n_com))
    l = 0.014*np.random.uniform(0.9,1.1,(2,n_com))
    plt.show()
    # check whether multispecies_equi does a good job
    equi,unfixed = multispecies_equi(phi/l,k_spec)
    
    I_in = lambda t: I_in_def(40)
    # solve with odeint
    def multi_growth(N_r,t):
        N = N_r.reshape(-1,n_com)
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = np.einsum("sc,lsc->lc", N, k_spec)[:,np.newaxis]
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
        return (N*(growth-l)).flatten()
        
    sols = odeint(multi_growth, equi.reshape(-1), np.linspace(0,100,10))
    print("Solution given by odeint: ",sols[-1].reshape(-1,2))
    print("Solution by homemade function:",equi)
    
    
    
    
    