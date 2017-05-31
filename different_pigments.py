# -*- coding: utf-8 -*-
"""
This file is to find the amount of different pigments that exit in nature
To understand the mathematical equations look at \4_analysis, 
Quantification of lightspectrum, pigments.docx
Assumptions: z_m = 1, Sum(\alpha_i)=1
"""

import numpy as np


from scipy.integrate import simps
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

import help_functions_chesson as ches

def pigments_distance(pigs, ratio, relfit,I_in = None, approx = True):
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
    f1 = 10**6/0.014*np.ones(relfit.shape)    #fitnes of resident species
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
    for i in range(50):
        # N*fit
        N_fit = np.einsum('rf,f->rf', equi,fit)
        #compute the function values
        y_simps = iterator(N_fit, ratio, x_simps)
        #int(iterator, dlambda, 400,700)
        equi = simps(y_simps, dx = dx)
    if np.amin(equi)<1:
        print("Warning, not all resident species have a equilibrium "+
              "denity above 1")
        equi[equis<1] = np.nan  

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

   
def coex_hul(invade, ratio, fit):
    coex_trip = np.where(invade>1e-3)
    points = [[ratio[i], ratio[j], fit[k]] for i,j,k in zip(*coex_trip)]
    return ConvexHull(points)
    
fit = 2**np.linspace(-1,1,11)
ratio = np.linspace(0,1,10)
equis = equilibrium([ches.k_green, ches.k_red], ratio,10**6/0.014*np.ones(len(fit)))
print(timer()-start)
a = pigments_distance([ches.k_green, ches.k_red], ratio,fit,approx = False)
print(timer()-start) 

fig = plt.figure()
Y,X = np.meshgrid(fit, ratio)
ax = fig.gca(projection = '3d')
ax.plot_surface(X,Y,equis)
fig = plt.figure()
ax = fig.gca()
print(np.amin(a))

plt.imshow(a[0], interpolation = 'bicubic')
plt.colorbar()
coex_trip = np.where(a>0)
points = [[ratio[i], ratio[j], fit[k]] for i,j,k in zip(*coex_trip)]
xs = np.array([ratio[i] for i in coex_trip[0]])
ys = np.array([ratio[i] for i in coex_trip[1]])
zs = np.array([fit[i] for i in coex_trip[2]])
fig = plt.figure()
ax = fig.add_subplot(111,projection= '3d')
ax.scatter(xs,ys,np.log(zs),s = 1)

