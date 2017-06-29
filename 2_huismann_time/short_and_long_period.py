

import analytical_communities as com
import numpy as np
from scipy import stats
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d

"""variables occuring in many functions, are defined here for all functions:
P: is the period length, for which the light stays the same
If this is None, then P will be generated and returned
I_r: Is the range in which the incoming light fluctuates
species = [in, re]: means the first entry of species, should contain the
parameters of the invader species and the second the parameters of the resident
If this is None, then species will be generated and returned
spec, inv, res: Contains the parameters of one species.
spec means a general species
inv and res should only be called for invader/resident species. They do compute
correctly if called otherwise, but this computation is usually not needed
itera: the number of iterations in a numerical approximation

###########
coex_test generates a species and a P and computes the boundary growth of
both species with different approaches calling the following functions:
exact_r_i, indep_envi_r_i, lin_approx_r_i, mp_approx_r_i
All these functions compute r_i, each function assumes more approximations
Exact_r_I:
    exactely computes the r_i, can't be computed analytically
    Takes about 5 seconds (10 seconds for both r_j)
indep_envi_r_i:
    assumes, that W_P(T+1) and W_P(T) are uncorelated, can't be computed ana.
    Takes about 0.3 seconds (0.6 for both r_j)
lin_approx:
    assumes, that C and E can be approximated linearly. 
    Takes about 0.2 seconds
mp_approx_r_i:
    assumes that mp = mr*((1-e^(-lr*P))/(1+e^(-lr*P)))^0.5
    and mr = dW/dI (I_av)
    takes 0.01 seconds

All these functions return r_i/P
    
The times are given for low P, for large P, the time can be reduced by the time
of dist_resident, which is about 0.15"""

    
def coex_test(species = None, P = None, exact = False, indep_envi = False,
              lin = True, mp = False):
    """ calls different functions to compute r_i
    
    if species and P are not given, they are generate by the function"""
    ret = []
    if species is None:
        species = com.find_species()
        ret.append(species)
    if P is None:
        P = np.random.uniform(0.1,10)
        ret.append(P)
    spec1, spec2 =list(species[0]), list(species[1])
    species_redo = np.array([spec2, spec1])
    functions = [exact_r_i,indep_envi_r_i,lin_approx_r_i, mp_approx_r_i]
    compute = [exact, indep_envi, lin, mp]
    for i in range(4):
        if compute[i]:
            comp1, stor1, r_i_1 = functions[i](species, P)
            comp2, stor2, r_i_2 = functions[i](species_redo, P)
            ret.extend([comp1, stor1, r_i_1, comp2, stor2, r_i_2])
    return (*ret,)
    
def exact_r_i(species,P, I_r = np.array([50,200])):
    """numerically computes the boundary growth rate of species[0]
    assumptions: I_out = 0"""
    comp, comp_fun = exact_comp(species, P, I_r) #delta C
    envi_fun = exact_envi(species, P, I_r)
    stor = exact_storage(species,comp_fun,envi_fun, P,I_r)/P #Delta I
    return  comp, stor,stor-comp

def exact_comp(species,P, I_r):
    """returns C(I)"""
    abs0 = lambda I: species[0][0]*com.equilibrium(I, species[0])
    abs1 = lambda I: species[1][0]*com.equilibrium(I, species[1])
    comp_fun = lambda I: abs0(I)/abs1(I)-1 #the competition function
    return -species[0][-1]*numerical_comp(comp_fun),comp_fun #Delta C

def exact_envi(species, P, I_r):
    """returns a function, that gives the distribution of the species densities
    over time"""
    Iav = np.average(I_r) #average incoming light
    #minimal, maximal and average density of the resident
    Wm, WM, Wav = [com.equilibrium(I, species[1]) for I in [*I_r, Iav]]    
                   
    epsilon = (WM-Wm)*np.exp(-P*species[1][-1])/Wm
    if epsilon > 0.01: #Period is not long enough to find new treshhold
        return dist_resident(species[1], Wav,P=P, I_r=I_r, interpolate = 'cubic')
    else:
        return lambda I: com.equilibrium(I, species[1])
        
def exact_storage(species, comp_fun, envi_fun, P,I_r = np.array([50,200])):
    """numerically integrates the integrals needed for the storage effect"""
    l = species[:,-1]
    def fun1(I_n, I_p):
        equi = com.equilibrium(I_n, species[1])
        end_density = equi-(equi-envi_fun(I_p))*np.exp(-l[1]*P)
        return comp_fun(I_n)*np.log(end_density/envi_fun(I_p))
    Im, IM = I_r
    Im_fun = lambda I: Im
    IM_fun = lambda I: IM    
    stor = dblquad(fun1,Im,IM, Im_fun, IM_fun)[0]    
    prefactor = l[0]/l[1]/(I_r[1]-I_r[0])**2 #contains gamma    
    return prefactor*stor 

###############################################################################
    
def indep_envi_r_i(species,P, I_r = np.array([50,200])):
    """numerically computes the boundary growth rate of species[0]
    additional assumptions: W_P(T+1) and W_P(T) are uncorelated
    """
    comp, comp_fun = exact_comp(species, P, I_r)
    envi_fun = exact_envi(species, P, I_r)
    stor = indep_storage(species,comp_fun,envi_fun, I_r)/P #only difference to exact
    return  comp, stor,stor-comp
    
def indep_storage(species, comp_fun, envi_fun, I_r = np.array([50,200])):
    """numerically integrates the integrals needed for the storage effect
    assumes, that W_P(T+1) and W_P(T) are uncorelated"""
    fun1 = lambda I: comp_fun(I)*np.log(envi_fun(I))
    save1 = quad(comp_fun, *I_r)[0] #integrals can be splited, as the are independent
    save2 = quad(lambda I: np.log(envi_fun(I)), *I_r)[0]
    save3 = quad(fun1, *I_r)[0]*(I_r[1]-I_r[0])
    l = species[:,-1]
    prefactor = l[0]/l[1]/(I_r[1]-I_r[0])**2 #contains gamma    
    return prefactor*(save3-save1*save2)

###############################################################################
    
def lin_approx_r_i(species, P, I_r = np.array([50,200])):
    """analytically computes the boundary growth rate of species[0]
    assumes, that com_fun, envi_fun are linear functions"""    
    Iav = np.average(I_r)
    mr,qr = linear_envi(species[1], I_r) #mr*I+qr = envi_fun
    
    Wm, WM = mr*I_r+qr
    Wav = mr*Iav+qr
    
    epsilon = (WM-Wm)*np.exp(-P*species[1][-1])/Wm
    if epsilon > 0.01: #Period is not long enough to find new treshhold 
        dist_res = dist_resident(species[1], Wav,P=P, I_r=I_r)
        Wm,WM = dist_res(I_r)
        
    mc, qc = linear_comp(species) # mc*I+qc-1 = comp_fun
    compe = -species[0][-1]*(Iav*mc+qc-1) # integral is just the average
    

    stor_prov = WM**2-Wm**2-2*WM*Wm*np.log(WM/Wm)
    l = species[:,-1]
    prefactor = l[0]/l[1]*mc/(4*mr*(WM-Wm))  #contains gamma                 
    stor = prefactor*stor_prov/P
    return  compe, stor, stor-compe
    
###############################################################################    
def mp_approx_r_i(species, P, I_r):
    k,H,p,l = species
    I_av = (I_r[:,0]+I_r[:,1])/2
            
    E_envi = np.zeros(species.shape[1]) #environment is zero
    
    # competition
    mc, qc = linear_comp(species) # mc*I+qc-1 = comp_fun
    E_comp = -l[:,1]*(mc*I_av+qc-1)
    
    # storage effect
    mr = p[:,1]/(l[:,1]*k[:,1]*(H[:,1]+I_av)) #=dW/dI (I_av)
    Wav = com.equilibrium(species[:,:,1],I_av,False) #mr*(I-Iav)+Wav = envi_fun
    
    #computation of mp is the only difference to lin_approx                  
    mp = mr*((1-np.exp(-l[:,1]*P))/(1+np.exp(-l[:,1]*P)))**0.5
    Wm = mp*(I_r[:,0]-I_av)+Wav
    WM = mp*(I_r[:,1]-I_av)+Wav
    stor_prov = WM**2-Wm**2-2*WM*Wm*np.log(WM/Wm)
    prefactor = l[:,0]/l[:,1]*mc/(4*mr*(WM-Wm))  #contains gamma             
    E_stor = prefactor*stor_prov/P
    return  E_envi,E_comp,E_stor, E_envi-E_comp+E_stor
   
def dist_resident(res, W_av = None, P = 5, I_r = [50,200], itera = 10001,
                          interpolate = False, plot_bool = False):
    """gives a distribution of W_p(T)
    if interpolate is False, then the data is approximated linearly,
    otherwise the "exact" data is returned"""
    l = res[-1]
    W_r = np.zeros(itera)
    lights = np.random.uniform(I_r[0], I_r[1], itera)
    lights[0] = np.average(I_r)
    W_r[0] = com.equilibrium(lights[0], res)
    for i in range(itera-1):
        W_r_star = com.equilibrium(lights[i+1], res, True)
        W_r[i+1] = W_r_star-(W_r_star-W_r[i])*np.exp(-l*P) #density at end of period
    if plot_bool:
        plot_percentile(W_r)
       
    adap_light = sorted(lights)-np.average(I_r)    
    if interpolate:
        if not(itera%100 == 1):
            print("itera must be of the form n*10+1")
            return None
        fun = interp1d(np.linspace(*I_r, 11), sorted(W_r)[0::int(itera/10)],
                       interpolate)
        return fun
    else:
        W_adap =W_r -W_av 
        cov = sum(adap_light*sorted(W_adap))
        var = sum(adap_light**2)
        m = cov/var # linear regression with 0 intercept, enforcing that m*Iav +q = Wav
        q = W_av-m*np.average(I_r)
        return lambda I: m*I+q
        
def linear_comp(species, I_r = [50,200], itera = 151):
    """makes a linear regression ot the eq. densities of the species"""
    light_r = np.linspace(*I_r, itera)
    equis = com.equilibrium(species, light_r)

    kW = np.einsum('is,isl->isl', species[0], equis) #k*equis
    kW_div = np.einsum('il,il->il', kW[:,0,:], 1/kW[:,1,:])
    
    light_av = np.average(light_r)
    kW_av = np.average(kW_div, axis = 1)
    kW_center = kW_div - np.einsum('i,l->il',kW_av,np.ones(itera))

    Sxx = np.sum((light_r-light_av)**2)
    Sxy = np.einsum('l,il->i', (light_r-light_av),(kW_center))
    
    slope = Sxy/Sxx
    intercept = kW_av-slope*light_av
    return slope, intercept

def linear_envi(spec, I_r = [50,200], itera = 151):
    """ gives a linear regression for species equilibrium densities and incoming light
    species is only one species"""
    light_r = np.linspace(*I_r, 151)
    W = np.array([com.equilibrium(i, spec, True) for i in light_r])
    m, q, r, p, st = stats.linregress(np.linspace(*I_r, 151),W)
    if r**2< 0.95 or p>10**(-3):
        print("possibly bad linear approximation",r**2,p)
    return m,q

def numerical_comp(comp_fun, I_r = [50,200]):
    """computes \DeltaC for different approximations of comp"""
    return quad(comp_fun,*I_r)[0]/(I_r[1]-I_r[0])
   
def r_i_period(species, I_now, W_res_prev, P = 5):
    """computes how much the invader species grows in this period"""
    W_now = com.equilibrium([I_now, I_now], species, True)
    C = species[1][0]*W_now[1]/(species[0][0]*W_now[0])
    E = np.log(W_now[1]/W_res_prev)
    return (1-C)/C*species[0][-1]*P+species[0][-1]/species[1][-1]*E/C