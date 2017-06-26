# -*- coding: utf-8 -*-
"""
This file is to find the amount of different pigments that exit in nature
To understand the mathematical equations look at \4_analysis, 
Quantification of lightspectrum, pigments.docx
Assumptions: z_m = 1, Sum(\alpha_i)=1
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from scipy.spatial import ConvexHull
from timeit import default_timer as timer
from scipy.interpolate import interp1d

import help_functions_chesson as ches

def multispecies_equi_single(pigs, fits):
    equis = 1e20*np.ones(len(fits)) # start of iteration
    lam, dx = np.linspace(400,700,101, retstep  = True) # points for integration
    
    # k_j(lam), shape = (len(fits), len(lam))
    abs_points = np.array([pig(lam) for pig in pigs])
    for i in range(5000):
        equis_old = equis.copy()
        # sum_j(N_j*k_j(lam)), shape = (len(lam),)
        tot_abs = np.einsum('n,nl->l', equis, abs_points)
        # N_j*k_j(lam), shape = (len(fits), len(lam))
        all_abs = np.einsum('n,nl->nl', equis, abs_points)

        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(len(fits), len(lam))
        y_simps = np.einsum('nl,l->nl', all_abs, (1-np.exp(-tot_abs))/tot_abs)
        # fit*int(y_simps)
        equis = fits*simps(y_simps, dx = dx)
        equis[equis<1] = 0

    return equis, equis_old
    
def multispecies_equi_randfit(pigs, itera = int(1e4),runs = 100, av_fit =1.4e8,
                        pow_fit = 2, per_fix = True, sing_pig = True):
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
        if i%10==0: #prograss report
            print(i)
            
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
        print("percent of fixed species:", np.sum(stable)/stable.size)
    equis = stable*equis
    
    #group the species that belong into one community
    return np.array([equis[i].reshape(-1) for i in range(len(equis))]).T
    
def plt_coex(equis):
    spec_num = [np.count_nonzero(i) for i in equis]
    plt.plot(np.linspace(0,100, len(spec_num)),sorted(spec_num))
    
def random_pigment_generator(n):
    """randomly generates `n` absorption spectra
    
    n: number of absorption pigments to create
    
    Returns: pigs, a list of functions. Each function in pigs (`pigs[i]`) is
    the absorption spectrum of this pigment.
    `pig[i](lam)` = sum_peak(gamma*e^(-(lam-peak)**2/sigma)"""
    #number of peaks for each pigment:
    npeak = 2+np.random.randint(5,size = n)
    # location of peaks
    peak = [np.random.uniform(400,700,(1,npeak[i])) for i in range(n)]
    # shape of peack
    sigma = [np.random.uniform(50,800, size = (1,npeak[i])) for i in range(n)]
    # magnitude of peak
    gamma = [np.random.uniform(0,1, size = (1,npeak[i])) for i in range(n)]
    pigs = []
    for i in range(n):
        pig = lambda lam, i = i: np.sum(gamma[i]*
            np.exp(-(lam.reshape(lam.size,1)-peak[i])**2/sigma[i]),axis = 1)
        pigs.append(pig)
    return pigs

def pigments_distance(pigs, ratio, relfit,I_in = None, approx = False
                      ,avefit = 1.36e8):
    """checks invasion possibility of a species with given pigs, ratio and fit
    
    Parameters:
        pigs: [pig1, pig2], list of fucntions
            pig1 and pig2 are the absorption spectrum of these two pigments
        ratio: array, 0<ratio<1
            proportion of pig1 that the species has
        relfit: array
            relative fitness of the species, relfit = fit_2/fit_1
            
    Returns:
        coex: array, coex.shape = (len(ratio), len(ratio), len(relfit))
            coex[i,j,k] = True <=> species 2 can invade, with:
            resident has pig ratio ratio[i]
            invader has pig ratio ratio[j]
            relativefitnes of species 2 is relfit[k]
    
    indexes in Einstein sumconvention:
        x-> x_simps
        r-> ratios of resident
        i-> ratios of invader
        f-> fitness
        p-> pigments"""
    # absorption of the pigments
    k = lambda lam: np.array([pig(lam) for pig in pigs])
    if I_in is None:
        I_in = lambda lam: 40/300 #corresponds to constant light with lux = 40
    # vector of ratios for invader and resident
    r_inv = np.array([ratio, 1-ratio])
    r_res = np.array([ratio, 1-ratio])
    # values for simpson, uneven number chosen for simpson
    x_simps,dx = np.linspace(400,700,101, retstep = True)
    # sum(f*alpha_(p,i)*k_p(lambda)) summed over p
    numerator = np.einsum('px,pif->xif',k(x_simps),
                          np.einsum('pi,f->pif',r_inv,relfit))
    # sum(alpha_(p,r)*k_p(lambda)) summed over p
    denominator = np.einsum('px,pr->xr',k(x_simps),r_res)
    # (numerator/denominator -1)*I_in(lambda)
    rel_abs = (np.einsum('xif,xr->xirf',numerator,1/denominator)-1)*40/300

    if approx: # simplified version
        return simps(rel_abs, dx = dx, axis = 0)
    f1 = avefit/np.sqrt(relfit)    #fitnes of resident species
    equis = equilibrium([ches.k_green, ches.k_red], ratio,f1)#equilibrium of res
    # (N_equi*f1)*denominator
    expon = np.einsum('rf,xr->xrf',np.einsum('rf,f->rf', equis,f1),denominator)
    # rel_abs*(1-np.exp(-expon))
    y_simps = np.einsum('jkil,jil->jkil',rel_abs, 1-np.exp(-expon))
    invade = simps(y_simps, dx = dx, axis = 0) #integrate
    return invade
    

zm = 1
I_in = lambda x:40/300*np.ones(shape = x.shape)
start = timer()
def equilibrium(pigs, ratio, fit):
    """Computes the equilibrium of the species in monoculture      
    
    Parameters:
        pigs: [pig1, pig2], list of fucntions
            pig1 and pig2 are the absorption spectrum of these two pigments
        ratio: array, 0<ratio<1
            proportion of pig1 that the species has
        fit: array
            fitness of the species, fit = phi/l
                
    Returns:
        equis: Array
            A twodim array, with equis.shape = (len(ratio),len(fit)) continaing
            the equilibrium of the species in monoculture.
            equis[i,j] = equilibrium(ratio[i], fit[j])
            
    indexes in Einstein sumconvention:
        x-> x_simps
        r-> ratios
        f-> fitness
        p-> pigments"""
    # values for simpson, uneven number chosen for simpson
    x_simps,dx = np.linspace(400,700,101, retstep = True)
    # absorption, sum(pig_i(lam)*alpha_i)
    absor = lambda ratio, lam: np.einsum('pr,px->rx',[ratio, 1-ratio],
                            np.array([pig(lam) for pig in pigs]))
    # function to be integrated I_in*(1-e^-N*f*absor(lambda))
    iterator = lambda N_fit, ratio, lam: 40/300*(1-np.exp(-
                        np.einsum('rf,rx->rfx',N_fit,absor(ratio, lam))))
    #iteratively search for equilibrium, start at infinity
    equi = np.infty*np.ones([len(ratio), len(fit)])
    for i in range(25):
        # N*fit
        N_fit = np.einsum('rf,f->rf', equi,fit)
        #compute the function values
        y_simps = iterator(N_fit, ratio, x_simps)
        #int(iterator, dlambda, 400,700)
        equi = simps(y_simps, dx = dx)
    if np.amin(equi)<1:
        print("Warning, not all resident species have a equilibrium "+
              "denity above 1")
        equi[equi<1] = np.nan  

    return equi
    
from scipy.integrate import quad
def check_pig_dif(ratios, relfit):
    """check whether pigments distance is done correct"""
    pigs = [ches.k_green, ches.k_red]
    equi = equilibrium(pigs, np.array([ratios[0]]), np.array([10**6/0.014]))[0,0]
    fun = lambda lam: 40/300*(1-np.exp(-equi*10**6/0.014*(ratios[0]*pigs[0](lam)+\
                                    (1-ratios[0])*pigs[1](lam))))
    if (quad(fun, 400,700)[0]-equi)/equi>0.001:
        print("equi computed wrong")
        return None
        
    numer = lambda lam: (relfit*(ratios[1])-ratios[0])*pigs[0](lam)+\
                (relfit*(1-ratios[1])-(1-ratios[0]))*pigs[1](lam)
    denom = lambda lam: ratios[0]*pigs[0](lam)+\
                                    (1-ratios[0])*pigs[1](lam)
    inte = lambda lam: numer(lam)/denom(lam)*fun(lam)
    diff_check = quad(inte,400,700)[0]
    diff = pigments_distance(pigs, np.array(ratios), np.array([relfit]), approx = False)
    return diff[1,0]-diff_check<0.01
    
def load_pig(n):
    pig_data = (np.genfromtxt("../2_Data/myfile2.csv", delimiter = ',').T)[n]
    lams = np.linspace(400,700,151)
    return lambda lam: 10**-7*interp1d(lams, pig_data)(lam)

funs = [load_pig(i) for i in range(29)]