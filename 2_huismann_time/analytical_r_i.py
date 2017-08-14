import analytical_communities as com
import numpy as np

"""
#example:
spec, P, I_r = com.generate_com(1000)
k,p,H,l = spec

E1,comp1, stor1, r_i1 =  exact_r_i(spec, P, I_r, itera = 1000)
E2,comp2, stor2, r_i2 =  lin_approx_r_i(spec, P, I_r)  
E3,comp3, stor3, r_i3 =  mp_approx_r_i(spec, P, I_r)
#print((r_i==r_i2).all())
plit(*r_i1)
plit(*r_i2)
plit(*r_i3)"""

def exact_r_i(species, P, I_r, itera = 20000):
    """ computes the exact boundariy growthrates of the species
    
    Parameters:
    #################
    species, P, I_r: As in output of com.generate_com
        species: species parameters
        P: the period length
        I_r: Light range
    itera: int, optional
        number random measurements for finding r_i
    
    Returns:
    #################
    E_envi: array, E_envi.shape = species[0].shape
        Average of environemental force
    E_comp: array, E_comp.shape = species[0].shape
        Average of competition
    E_stor: array, E_stor.shape = species[0].shape
        Storage effect
    r_i: array, r_i.shape = species[0].shape
        Sum of the above
    Important: All results are divided by the period length P"""
    
    I_av = (I_r[0]+I_r[1])/2 #average = balance point of I_in
    Wav = com.equilibrium(species,I_av,'simple') # density at average i_in
    
    # W_r_ef[:,i,j] is a serie of densities for species i, replicate j
    # similar for W_r_eq, W_r_eq[i] corresponds to lights[i]
    W_r_eq, W_r_ef, lights = dist_resident(species,  P, I_r, Wav, False, itera)

    W_i_eq = W_r_eq.copy() #contains invader densities at the same lights
    W_i_eq[:,0], W_i_eq[:,1] = W_r_eq[:,1], W_r_eq[:,0]
    
    #species parameters of the resident
    kr,lr = species[[0,-1]]
    #invader values, are flipped or resident
    li = np.array([lr[1], lr[0]]) 
    ki = np.array([kr[1], kr[0]])
    
    # environmental parameter for one period
    E = np.log(W_r_ef[1:]/W_r_ef[:-1])
    
    # competition in one period, remove first, as it is identical to last
    C = (kr*W_r_eq/(ki*W_i_eq))[1:]

    # take average, =0
    E_envi = li/lr*np.average(E, 0)  
    E_comp = -li*P*(np.average(1/C, axis = 0)-1)

    # E_stor = np.average(li/lr*E, -li*P*(1/C-1)), gamma = -1/(li*P)
    E_stor = li/lr*np.average(E/C,0)-E_envi #include gamma immediately
    #return, divide by P to compare different period length
    return E_envi/P, E_comp/P, E_stor/P, (E_envi-E_comp+E_stor)/P

def lin_approx_r_i(species, P, I_r = np.array([50,200])):    
    """ Computes the boundary growth rate with an liniear approximation of C, E
    
    Is identical to mp_approx, except the computation of Wm,WM
    Parameters and return same as exact_r_i"""
    k,H,p,lr = species #parameters of the species
    li = np.array([lr[1],lr[0]]) #invader value
    I_av = (I_r[0]+I_r[1])/2 #average = balance point of I_in

    # E(environment)
    E_envi = np.zeros(species[0].shape) 
    
    # E(competition)
    mc,qc = linear_comp(species, I_r) # 1/C = mc*I+qc-1 

    E_comp = -li*P*(mc*I_av+qc-1) #l*int(1/C-1)
    
    # storage effect
    mr = p/(lr*k*(H+I_av)) #=dW/dI (I_av)
    Wav = com.equilibrium(species,I_av,'simple') # mr*(I-Iav)+Wav = envi_fun

    ######### Difference to mp_lin_approx
    # Computation of mp, check whether mp = mr
    Wm = com.equilibrium(species, I_r[0], 'simple')
    WM = com.equilibrium(species, I_r[1], 'simple')
    
    #epsilon < 0.01-> resident will always find it's new equilibrium                     
    epsilon = (WM-Wm)*np.exp(-P*lr)/Wm
    short_period = np.logical_or(*(epsilon>0.005))
    mp = mr.copy()
    # find actual distributio of W, get mp
    mp[:,short_period] = dist_resident(species[:,:,short_period], 
                P[short_period], I_r[:,short_period], Wav[:,short_period])
    ########## again identical to mp_approx_r_i
    
    
    # maximal and minimal densities reached
    Wm = mp*(I_r[0]-I_av)+Wav 
    WM = mp*(I_r[1]-I_av)+Wav
    
    # storage effect, see LCH2, Assumption 1, long periods, Assumption 1.1
    stor_prov = WM**2-Wm**2-2*WM*Wm*np.log(WM/Wm)
    
    prefactor = li/lr*mc/(4*mr*(WM-Wm))  #contains gamma             
    E_stor = prefactor*stor_prov/P
    #return all the elements
    return  E_envi,E_comp/P,E_stor, E_envi-E_comp/P+E_stor
    
def mp_approx_r_i(species, P, I_r):
    """ Computes the boundary growth rate with an approximation of mp
    
    Is identical to lin_approx, except the computation of Wm,WM
    Parameters and return same as exact_r_i"""
    k,H,p,lr = species #parameters of the species
    li = lr[::-1] #invader values
    I_av = (I_r[0]+I_r[1])/2 #average = balance point of I_in

    # E(environment)
    E_envi = np.zeros(species[0].shape)
    
    # E(competition)
    mc,qc = linear_comp(species, I_r) # 1/C = mc*I+qc-1 

    E_comp = -li*P*(mc*I_av+qc-1) #l*int(1/C-1)
    
    # storage effect
    mr = p/(lr*k*(H+I_av)) #=dW/dI (I_av)
    Wav = com.equilibrium(species,I_av,'simple') # mr*(I-Iav)+Wav = envi_fun


    # mp is the propotion of Wm that is reached,only diff to lin_approx
    mp = mr*((1-np.exp(-lr*P))/(1+np.exp(-lr*P)))**0.5
    
    # maximal and minimal densities reached
    Wm = mp*(I_r[0]-I_av)+Wav 
    WM = mp*(I_r[1]-I_av)+Wav
    
    # storage effect, see LCH2, Assumption 1, long periods, Assumption 1.1
    stor_prov = WM**2-Wm**2-2*WM*Wm*np.log(WM/Wm)
    
    prefactor = li/lr*mc/(4*mr*(WM-Wm))  #contains gamma             
    E_stor = prefactor*stor_prov/P
    #return all the elements
    return  E_envi,E_comp/P,E_stor, E_envi-E_comp/P+E_stor

def dist_resident(spec,P, I_r, W_av,  lin = True, itera = 20000):
    """gives a distribution of W_p(T)
    
    Parameters:
    #################
    species, P, I_r: As in output of com.generate_com
        species: species parameters
        P: the period length
        I_r: Light range
    W_av: array, W_av.shape = spec[0].shape
        equilibrium density at average incoming light
    lin: Boolean, optional
        If True, linear regression of data is returned
    itera: int, optional
        number of randomly generated lights
    
    Returns:
    #################
    Depends on the parameter lin:
    if lin==True:
        slope of linear regression of effective densities for random light
        sequence
    if lin==False:
        W_r_eq: array, shape = (itera,)+spec[0].shape
            equilibrium densities of species at different lights
        W_r_ef: array, same shape as W_r_eq
            effective densities of species at different lights
        lights: array, shape = (itera, len(spec[0,0,:]))
            lights used for the equilibrium densities"""
    l = spec[-1]
    # relative distribution of the lights, same for all species
    rel_lights = np.random.uniform(size = (itera,1))
    # linear transformation for all communities
    lights = I_r[0]+(I_r[1]-I_r[0])*rel_lights
    lights[0] = (I_r[0]+I_r[1])/2 # same starting and ending light
    lights[-1] = (I_r[0]+I_r[1])/2
    
    #compute the equilibria
    W_r_eq = com.equilibrium(spec,lights, 'partial')
    # iteratively find the effective densities, start at average density
    W_r_ef = W_av*np.ones(itera).reshape((-1,1,1))
    lP = (l*P).reshape((1,)+l.shape) # how fast species find new density
    # iteratively find new effective density
    for i in range(5):
        W_r_ef[1:] = W_r_eq[1:]-(W_r_eq[1:]-W_r_ef[:-1])*np.exp(-lP)
    if lin: #return linear regression of data
        light_av = np.average(rel_lights)
        W_r_av = np.average(W_r_ef, axis = 0)

        W_r_center = W_r_ef - W_r_av
        # variance in light
        Sxx = np.sum((rel_lights-light_av)**2)**2
        # covariance of light and equilbria
        Sxy = np.einsum('la,lsi->si', (rel_lights-light_av),(W_r_center))
        # return slope of linear regression                        
        return Sxy/(Sxx*(I_r[1]-I_r[0]))
    else:
        return W_r_eq, W_r_ef, lights

def linear_comp(species, I_r = [50,200], itera = 151):
    """makes a linear regression ot the eq. densities of the species
    
    Parameters:
    ##############
    species, I_r as the output of com.generate_communities
    itera: int
        precision for linear regression
    
    Returns
    ##############
    slope: array, slope.shape = species[0].shape
        slope of linear regression
    intercept: array, same shape as slope
        intercept of linear regression"""
    rel_lights = np.linspace(0,1,itera) #uniform distribution of lights
    # variable transformation, different lights for different communities
    lights = I_r[0]+(I_r[1]-I_r[0])*rel_lights[:,np.newaxis]
    #find equilibria for all species
    equis = com.equilibrium(species, lights, 'partial')
    k = species[0]
    kW = k*equis #absorption
    # kW_i/kW_r = 1/C
    kW_div = np.array([kW[:,1]/kW[:,0],kW[:,0]/kW[:,1]])

    #linear regression
    light_av = np.average(rel_lights)
    kW_av = np.average(kW_div, axis = 1)
    kW_centered = kW_div-kW_av[:,np.newaxis,:]
    
    # variance of lights
    Sxx = np.sum((rel_lights-light_av)**2)
    # covariance of rel_lights and the equilibria
    Sxy = np.einsum('l,sli->si', rel_lights-light_av,(kW_centered))
    
    # slope of linregress, divide by light range, as we have regressed with rel_lights    
    slope = Sxy/(Sxx*(I_r[1]-I_r[0]))
    intercept = kW_av-slope*np.average(I_r, axis = 0)
    return slope, intercept