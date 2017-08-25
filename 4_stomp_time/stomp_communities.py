"""
@author: J.W.Spaak

Generates random species

`gen_species` returns two random species

`sat_carbon_par`: rand. generate two species with saturating carbon uptake
    functions
`photoinhibition_par`: rand. generate two species with carbon uptake
    functions that suffer from photoinhibition
    
`equilibrium` computes the equilibrium density of the species


"""
import numpy as np
from scipy.integrate import simps
from timeit import default_timer as timer

def I_in_def(lux, loc = 550, sigma = 0):
    if sigma == 0:
        return lambda lam: np.full(lam.shape, lux)
    else:
        return lambda lam: lux*np.exp(-(lam-loc)**2/sigma)
        
        
def gen_species(num = 20, I_in1 = I_in_def(100/300), I_in2=I_in_def(25/300), richness = 3):
    found = 0
    itera = 1000
    equi = np.empty((2,3,itera))
    species_f = np.empty((2,richness,num))
    
    while found<num:
        start = timer()
        # generating the species
        species, k_spec, g_spec = stomp_par(num = itera, richness = 3)
        print(np.amax(species[1])/np.amin(species[1]))
        # computing the equilibrium densities
        equi[0] = equilibrium(k_spec, species, I_in1)
        equi[1] = equilibrium(k_spec, species, I_in2)
        # which ones did survive
        surv = equi>1e5
        # specues that invaded/went extinct
        rich_change = np.sum(np.logical_xor(surv[0], surv[1]),0)>1
        good =(np.sum(surv[0],0)==richness-1)&(np.sum(surv[1],0)==richness-1)\
                 & rich_change
        #species_f[:,:,found:found+np.sum(good)] = species[:,:,good]
        found += np.sum(good)
        print(found, rich_change.sum(), timer()-start)
    return
        
    

def stomp_par(num = 1000,richness = 3, fac = 10):
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
    pigments = random_pigments(num, richness-1, fac)
    # raise by
    pig_ratio = np.random.beta(0.1,0.1, (richness-1,richness, num))
    k_spec = lambda lam: np.einsum('psn,lpn->lsn', pig_ratio, pigments(lam))
    k_spec.pigments = pigments
    k_spec.pig_ratio = pig_ratio
    
    # carbon uptake fucntion
    lambs = np.linspace(400,700,31)
    I_in = lambda lam: 40/300*np.ones(lam.shape)
    def g_spec(density,t,species, k_spec, I_in = I_in,lambs =lambs):
        ncom = species.shape[-1]
        # k_i(lam)*N_i, shape = len(lambs), n_com
        absor = density*k_spec(lambs)
        tot_abs = np.sum(absor, axis = 1) # \sum_i  k_i(lam)*N_i
        rel_absor = absor/tot_abs.reshape(-1,1,ncom) #relative absorption
        
        I_in = I_in(lambs).reshape(-1,1)
        # integrate_400^700 rel_absor*I_in(lambda)*(1-e^(-absorption)) dlambda
        energy = I_in*(1-np.exp(-tot_abs))
        dx = lambs[1]-lambs[0]
        energy = simps(rel_absor*energy.reshape(-1,1,ncom),dx = dx, axis = 0)
        # phi*integral(absorbtion)- l*N
        return species[0]*energy-species[1]*density
    
    return np.array([phi, l]), k_spec, g_spec
    
    
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
    
def equilibrium(k_spec, species, I_in = lambda lam: 40/300*np.ones(lam.shape)
                    ,runs = 100):
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
    
    # asign a fitness to all species, shape = (npi, itera)
    fitness = species[0]/species[1]
    
    # starting densities for iteration, shape = (npi, itera)
    equis = np.full(species[0].shape, 1e10) # start of iteration
    equis_fix = np.zeros(equis.shape)

    lam, dx = np.linspace(400,700,51, retstep = True) #points for integration
    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = k_spec(lam)
    I_in = I_in(lam)[:,np.newaxis]
    unfixed = np.full(species.shape[-1], True, dtype = bool)
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
    # species not found equilibrium are not considered further        
    equis_fix[:,unfixed] = np.nan 
    return equis_fix
    
#gen_species()
"""
species, k_spec, g_spec = stomp_par(num = 1000, richness = 3)
# computing the equilibrium densities
equi = equilibrium(k_spec, species, I_in_def(40/300))
print("fine")
start = timer()
#surv = gen_species()
print(timer()-start)"""
"""
plt.plot(np.linspace(400,700,151),k_spec(np.linspace(400,700,151))[:,:,0])
plt.figure()
plt.plot(np.linspace(400,700,151),pigments(np.linspace(400,700,151))[:,:,0])"""