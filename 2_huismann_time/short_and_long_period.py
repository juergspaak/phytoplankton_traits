import analytical_communities as com
import numpy as np

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
    
def exact_r_i(species, P, I_r):
    I_av = (I_r[0]+I_r[1])/2 #average = balance point of I_in
    Wav = com.equilibrium(species[:,1],I_av,'simple')
    W_r_ef, W_r_eq, lights = dist_resident(species[:,1],  P, I_r, Wav, False)
    W_i_eq = com.equilibrium(species[:,0],lights, 'partial')
    k,l = species[[0,-1]]
    C = (k[1]*W_r_eq/(k[0]*W_i_eq))[1:]
    E = np.log(W_r_ef[1:]/W_r_ef[:-1]) 
    curl_C = -(species[-1,0]*P).reshape((1,-1))*(1/C-1)
    curl_E = (species[-1,0]/species[-1,1])*E
    E_envi = np.average(curl_E, 0)
    E_comp = np.average(curl_C,0)
    E_stor = -np.average(curl_E*curl_C,0)/(species[-1,0]*P)
    return E_envi/P, E_comp/P, E_stor/P, (E_envi-E_comp+E_stor)/P
    
def mp_approx_r_i(species, P, I_r):
    """ Computes the boundary growth rate with an approximation of mp
    
    Is identical to lin_approx, except the computation of Wm,WM
    See also lin_approx_r_i, exact_r_i"""
    k,H,p,l = species #parameters of the species
    I_av = (I_r[0]+I_r[1])/2 #average = balance point of I_in

    # E(environment)
    E_envi = np.zeros(species.shape[-1]) 
    
    # E(competition)
    mc, qc = linear_comp(species) # comp_fun = mc*I+qc-1 
    E_comp = -l[1]*(mc*I_av+qc-1) #l*int(comp_fun)
    
    # storage effect
    mr = p[1]/(l[1]*k[1]*(H[1]+I_av)) #=dW/dI (I_av)
    Wav = com.equilibrium(species[:,1],I_av,'simple') #mr*(I-Iav)+Wav = envi_fun
    
    #computation of Wm,WM is the only difference to lin_approx                  
    mp = mr*((1-np.exp(-l[1]*P))/(1+np.exp(-l[1]*P)))**0.5
    Wm = mp*(I_r[0]-I_av)+Wav #mp is the propotion of Wm that is reached
    WM = mp*(I_r[1]-I_av)+Wav
    
    stor_prov = WM**2-Wm**2-2*WM*Wm*np.log(WM/Wm)
    prefactor = l[0]/l[1]*mc/(4*mr*(WM-Wm))  #contains gamma             
    E_stor = prefactor*stor_prov/P
    
    #return all the elements
    return  E_envi,E_comp,E_stor, E_envi-E_comp+E_stor
    
def linear_comp(species, I_r = [50,200], itera = 151):
    """makes a linear regression ot the eq. densities of the species"""
    light_r = np.linspace(*I_r, itera)
    equis = com.equilibrium(species, light_r)
    
    kW = species[0].reshape((1,)+species[0].shape)*equis
    kW_div = kW[:,0]/kW[:,1]

    light_av = np.average(light_r)

    kW_av = np.average(kW_div, axis = 0)

    #kW_center = kW_div - np.einsum('i,l->li',kW_av,np.ones(itera))
    kW_centered = kW_div-kW_av.reshape((1,-1))

    Sxx = np.sum((light_r-light_av)**2)
    Sxy = np.einsum('l,li->i', (light_r-light_av),(kW_centered))
    
    slope = Sxy/Sxx
    intercept = kW_av-slope*light_av
    return slope, intercept

def lin_approx_r_i(species, P, I_r = np.array([50,200])):    
    """Computes the boundary growth rate, assumes com_fun, env_fun are linear
    
    Is identical to lin_approx, except the computation of Wm,WM
    See also lin_approx_r_i, exact_r_i"""
    k,H,p,l = species #parameters of the species
    I_av = (I_r[0]+I_r[1])/2 #average = balance point of I_in

    # E(environment)
    E_envi = np.zeros(species.shape[-1]) 
    
    # E(competition)
    mc, qc = linear_comp(species) # comp_fun = mc*I+qc-1 
    E_comp = -l[1]*(mc*I_av+qc-1) #l*int(comp_fun)
    
    # storage effect
    mr = p[1]/(l[1]*k[1]*(H[1]+I_av)) #=dW/dI (I_av)
    mp = mr.copy()
    Wav = com.equilibrium(species[:,1],I_av,'simple') #mr*(I-Iav)+Wav = envi_fun
    Wm = com.equilibrium(species[:,1], I_r[0], 'simple')
    WM = com.equilibrium(species[:,1], I_r[1], 'simple')
    
    #epsilon < 0.01-> resident will always find it's new equilibrium                     
    epsilon = (WM-Wm)*np.exp(-P*l[1])/Wm
    short_period = epsilon>0.005
    
    mp[short_period] = dist_resident(species[:,1,short_period], 
                         P[short_period], I_r[:,short_period], Wav[short_period])    
    
    Wm = mp*(I_r[0]-I_av)+Wav #mp is the propotion of Wm that is reached
    WM = mp*(I_r[1]-I_av)+Wav
    
    stor_prov = WM**2-Wm**2-2*WM*Wm*np.log(WM/Wm)
    prefactor = l[0]/l[1]*mc/(4*mr*(WM-Wm))  #contains gamma             
    E_stor = prefactor*stor_prov/P
    
    #return all the elements
    return  E_envi,E_comp,E_stor, E_envi-E_comp+E_stor
    
def dist_resident(res,P, I_r, W_av,  lin = True, itera = 20001):
    """gives a distribution of W_p(T)
    if interpolate is False, then the data is approximated linearly,
    otherwise the "exact" data is returned"""
    l = res[-1]
    rel_lights = np.random.uniform(size = (itera,1))
    lights = I_r[0]+(I_r[1]-I_r[0])*rel_lights
    
    lights[0] = (I_r[0]+I_r[1])/2
    lights[-1] = (I_r[0]+I_r[1])/2
    W_r_eq = com.equilibrium(res,lights, 'partial')
    W_r_ef = W_av*np.ones(itera).reshape((-1,1))
    lP = (l*P).reshape((1,-1))
    for i in range(5):
        W_r_ef[1:] = W_r_eq[1:]-(W_r_eq[1:]-W_r_ef[:-1])*np.exp(-lP)
    if lin: #return linear approximation of data
        light_av = np.average(rel_lights)
        W_r_av = np.average(W_r_ef, axis = 0)

        W_r_center = W_r_ef - W_r_av

        Sxx = np.sum((rel_lights-light_av)**2)*(I_r[1]-I_r[0])**2
        Sxy = np.einsum('la,li->i', (rel_lights-light_av),(W_r_center))\
                                *(I_r[1]-I_r[0])
        return Sxy/Sxx
    else:
        return W_r_ef, W_r_eq, lights

