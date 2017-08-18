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
    pigments = random_pigments(num, richness-1, fac)
    # raise by
    pig_ratio = np.random.beta(0.1,0.1, (richness-1,richness, num))
    pig_ratio = pig_ratio/np.sum(pig_ratio, axis = 0)
    k_spec = lambda lam: np.einsum('psn,lpn->lsn', pig_ratio, pigments(lam))
    
    # carbon uptake fucntion
    lambs = np.linspace(400,700,31)
    I_in = lambda lam: 40/300*np.ones(lam.shape)
    def g_spec(density,t,species, k_spec, I_in = I_in,lambs =lambs):
        ncom = species.shape[-1]
        # k_i(lam)*N_i, shape = len(lambs), n_com
        absor = density*k_spec(lambs)
        tot_abs = np.sum(absor, axis = 1) # \sum_i  k_i(lam)*N_i
        rel_absor = absor/tot_abs.reshape(-1,1,ncom) #relative absorption
        if False and np.isnan(tot_abs).any():
            a = np.isnan(tot_abs)
            print(absor[0,:,a[0]])
            print(density[:,a[0]], k_spec(lambs)[0,:,a[0]])
            
        I_in = I_in(lambs).reshape(-1,1)
        # integrate_400^700 rel_absor*I_in(lambda)*(1-e^(-absorption)) dlambda
        energy = I_in*(1-np.exp(-tot_abs))
        dx = lambs[1]-lambs[0]
        energy = simps(rel_absor*energy.reshape(-1,1,ncom),dx = dx, axis = 0)
        #return phi*integral(absorbtion)- l*N
        return species[0]*energy-species[1]*density
    
    return np.array([phi, l]), k_spec, g_spec, pigments, pig_ratio
    
    
def random_pigments(num, richness, n_peak_max = 3):
    """randomly generates `n` absorption spectra
    
    n: number of absorption pigments to create
    
    Returns: pigs, a list of functions. Each function in pigs (`pigs[i]`) is
        the absorption spectrum of this pigment.
    `pig[i](lam)` = sum_peak(gamma*e^(-(lam-peak)**2/sigma)"""
    shape = (n_peak_max, richness, num)
    
    # number of peaks for each pigment:
    n_peaks = np.random.binomial(1,0.5,shape)
    n_peaks[0] = 1 #each phytoplankton has at least 2 peaks
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
    return pigments
    
def equilibrium(k_spec, species, I_in = lambda lam: 40/300*np.ones(lam.shape)
                    ,runs = 200,):
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
    equis = np.full(species[0].shape, 1e20) # start of iteration
    
    lam, dx = np.linspace(400,700,101, retstep = True) #points for integration
    # k_j(lam), shape = (len(fits), len(lam))
    abs_points = k_spec(lam)
    print(abs_points.shape)
    I_in = I_in(lam)
    for i in range(runs):
        if i == runs-1:
            #save previous values in final run to see whether equilibrium has 
            #been reached
            equis_old = equis.copy()
        if i%(int(runs/10))==0: #prograss report
            print(100*i/runs, "percent done")
            
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = np.einsum('ni,lni->li', equis, abs_points)
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = equis*abs_points #np.einsum('ni,li->nli', equis, abs_points)
        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = all_abs*(1/tot_abs*I_in[:,np.newaxis]*(1-np.exp(-tot_abs)))[:,None]
        #y_simps = np.einsum('nli,li->nli', 
        #                    all_abs, (1-np.exp(-tot_abs))/tot_abs)
        # fit*int(y_simps)
        equis = fitness*simps(y_simps, dx = dx, axis =0)
        # remove rare species
        equis[equis<1] = 0
    # exclude the species that have not yet found equilibrium, avoid nan
    stable = np.logical_or(equis == 0,(equis-equis_old)/equis_old<0.0001)
    if True:
        print("percent of species that reached equilibrium:",
              np.sum(stable)/stable.size)
    equis = stable*equis
    
    #group the species that belong into one community
    return np.array([equis[i].reshape(-1) for i in range(len(equis))]).T
    
species, k_spec, g_spec, pigments, pratio = stomp_par(richness = 5)
print("fine")

"""
plt.plot(np.linspace(400,700,151),k_spec(np.linspace(400,700,151))[:,:,0])
plt.figure()
plt.plot(np.linspace(400,700,151),pigments(np.linspace(400,700,151))[:,:,0])"""