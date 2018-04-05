"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Contains functions that generate coexisting communities with changing incoming
light
"""
import numpy as np
from scipy.integrate import simps

from load_pigments import lambs, dlam
import multispecies_functions as mf
from generate_species import gen_com, n_diff_spe

from differential_functions_copy import own_ode
                    
def I_in_t(I_in1, I_in2, period):
    # returns light at given time t, fluctuates between I_in1, I_in2
    def fun(t):
        t_rel = (t%period)/period
        part_1 = 2*np.abs(t_rel-0.5)
        return part_1*I_in1+(1-part_1)*I_in2
    return fun
    
def find_survivors(equi, species_id):
    # compute the average amount of each species in the communities
    return [np.sum(species_id[equi>0] == i)/equi.shape[-1] 
                    for i in range(n_diff_spe)]

def pigment_richness(equi, alpha):
    return np.mean(np.sum(np.sum(equi*alpha, axis = 1)>0, axis = 0))

# standard incoming light fluctuation    
I_in_ref = I_in_t(mf.I_in_def(40/300,450,50), mf.I_in_def(40/300,650,50),10)

def fluctuating_richness(present_species = np.arange(5), r_spec = 10,
    n_com = 100,fac = 3,randomized_spectra = 0, l_period = 10,
    I_in = I_in_ref, t_const = [0,0.5]):
    """Computes the number of coexisting species
    
    Parameters:
    r_spec: int
        richness of species in the regional community
    n_com: int
        Number of communities to generate
    fac: float
        maximal factor by which traits of species differ
    l_period: float
        Lenght of period of fluctuating incoming light
    I_in: callable
        Must return an array of shape (101,). Incoming light at time t
    t_const: array-like
        Times at which species richness must be computed for constant light
    randomized_spectra: float
        amout by which pigments spectra differ from species
    allow_shortcut: boolean
        If True, when for all constant light cases the same species coexist,
        then fluctuating incoming light is not computed
    
    Returns:
    ret_mat: array, shape (len(t_const)+2, 10)
        ret_mat[i,j] Percentages of communities that have j coexisting species
        in the incoming light situation j. j in range(len(t_const)) means
        I_in(t_const[j]*period) as incoming light. j=len(t_const) is the 
        maximum of all constant incoming lights and the last one is the
        fluctuation incoming light
    intens_const:
        Intensity of outcoming light for the constant incoming light cases
    intens_fluct:
        Intensity of outcoming light for the fluctuating incoming light case"""
    
    ###########################################################################
    # find potentially interesting communities
             
    # generate species and communities
    par,k_spec,alpha,species_id = gen_com(present_species, r_spec, fac, n_com)
    # compute pigment richness at the beginning (the same in all communities)
    r_pig_start = pigment_richness(1, alpha)
    if randomized_spectra>0:
        # slightly change the spectra of all species
        # equals interspecific variation of pigments
        eps = randomized_spectra
        k_spec *= np.random.uniform(1-eps, 1+eps, k_spec.shape)
    phi,l = par
    
    # compute the equilibria densities for the different light regimes
    equi = np.empty((len(t_const),) + phi.shape)
    unfixed = np.empty((len(t_const),phi.shape[-1]))
    
    for i,t in list(enumerate(t_const)):
        equi[i], unfixed[i] = mf.multispecies_equi(phi/l, k_spec, 
                            I_in(t*l_period))
    # consider only communities, where algorithm found equilibria (all regimes)
    fixed = np.logical_not(np.sum(unfixed, axis = 0))
    equi = equi[..., fixed]
    phi = phi[:, fixed]
    l = l[:, fixed]
    k_spec = k_spec[..., fixed]
    species_id = species_id[...,fixed]
    alpha = alpha[...,fixed]

    ###########################################################################
    # return values for constant cases
    # richness in constant lights
    richness_equi = np.zeros(len(t_const)+1)
    richness_equi[:-1] = np.mean(np.sum(equi>0, axis = 1),axis = 1)
    
    surviving_species = np.zeros((len(t_const)+1,n_diff_spe))
    r_pig_equi = np.zeros(len(t_const)+1)
    for i in range(len(t_const)):
        surviving_species[i] = find_survivors(equi[i], species_id)
        r_pig_equi[i] = pigment_richness(equi[i], alpha)
    # Compute EF, biovolume
    EF_biovolume = np.full((len(t_const)+1,5),np.nan)
    EF_biovolume[:len(t_const)] = np.percentile(np.sum(equi, axis = 1),
                                [5,25,50,75,95], axis = -1).T
    
    EF_pigment = np.zeros(len(t_const)+1)
    EF_pigment[:len(t_const)] = np.mean(np.einsum("tsc,psc->tc",equi,alpha),
                                    axis = 1)

    
    ###########################################################################
    # Prepare computation for fluctuating incoming light
    # set 0 all species that did not survive in any of the cases
    dead = np.sum((equi>0), axis = 0)==0
    phi[dead] = 0
    l[dead] = 1 # to avoid division by 0
    k_spec[:,dead] = 0

    # maximal richness over all environments in one community
    max_spec = np.amax(np.sum(equi>0, axis = 1))
    # sort them accordingly to throw rest away
    com_ax = np.arange(equi.shape[-1])
    spec_sort = np.argsort(np.amax(equi,axis = 0), axis = 0)[-max_spec:]
    phi = phi[spec_sort, com_ax]         
    l = l[spec_sort, com_ax]
    equi = equi[np.arange(len(t_const)).reshape(-1,1,1),spec_sort, com_ax]
    k_spec = k_spec[np.arange(len(lambs)).reshape(-1,1,1),spec_sort, com_ax]
    species_id = species_id[spec_sort, com_ax]
    alpha = alpha[:,spec_sort, com_ax]
       
    ###########################################################################
    # take average densitiy over all lights for the starting density
    start_dens = np.mean(equi, axis = 0)

    def multi_growth(N,t,I_in, k_spec, phi,l):
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = np.einsum("sc,lsc->lc", N, k_spec)[:,np.newaxis]
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
        return N*(growth-l)
    
    n_period = 100 # number of periods to simulate in one run
    
    undone = np.arange(phi.shape[-1])
    # compute 100 periods, 10 timepoints per period
    time = np.linspace(0,l_period*n_period,n_period*10)
    phit,lt,k_spect = phi.copy(), l.copy(), k_spec.copy()
    # to save the solutions found
    sols = np.empty((10,)+phi.shape)
    
    # simulate densities
    counter = 1 # to avoid infinite loops

    while len(undone)>0 and counter <1000:
        sol = own_ode(multi_growth,start_dens, time[[0,-1]], 
                      I_in, k_spect, phit,lt,steps = len(time))
        
        # determine change in densities, av at end and after finding equilibria
        av_end = np.average(sol[-10:], axis = 0) 
        av_start = np.average(sol[-110:-100], axis = 0) 
        
        # relative difference in start and end
        rel_diff = np.nanmax(np.abs((av_end-av_start)/av_start),axis = 0)
        # communities that still change "a lot"
        unfixed = rel_diff>0.005
        
        # save equilibria found
        sols[...,undone] = sol[-10:]
        
        # select next communities
        undone = undone[unfixed]
        phit,lt,k_spect = phi[:, undone], l[:, undone], k_spec[...,undone]
        start_dens = sol[-1,:,unfixed].T
        # remove very rare species
        start_dens[start_dens<start_dens.sum(axis = 0)/5000] = 0
        counter += 1

    #######################################################################
    # preparing return values for richnesses computation
    EF_biovolume[-1] = np.percentile(np.sum(sols, axis = (0,1)),
                                [5,25,50,75,95], axis = -1)/len(sols)   
    # find number of coexisting species through time
    richness_equi[-1] = np.mean(np.sum(sols[-1]>0,axis = 0))
    surviving_species[-1] = find_survivors(sols[-1], species_id)
    r_pig_equi[-1] = pigment_richness(sols[-1], alpha)
    EF_pigment[-1] = np.mean(np.einsum("tsc,psc->c",sols,alpha))/len(sols)
    
    return (richness_equi, EF_biovolume, r_pig_equi, EF_pigment, r_pig_start,
            surviving_species)
            

if __name__ == "__main__":
    present_species = np.arange(5)
    r_pig = 5
    r_spec = 10
    r_pig_spec = 3
    n_com = 100
    fac = 3
    l_period = 10
    pigs = "real"
    I_in = I_in_ref
    t_const = [0,0.5]
    randomized_spectra = 0
    allow_shortcut = False