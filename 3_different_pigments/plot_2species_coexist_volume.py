"""
@author: J. W. Spaak, jurg.spaak@unamur.be

This file is to find the coexistence regions for two species and two pigments
"../../4_analysis/3.Finite pigments/3. Finite pigments.docx", 
Assumptions: z_m = 1, Sum(\alpha_i)=1

`equilibrium` computes the equilibrium density of one species in monoculture
    with different fitness and different abs. spectrum
    
`invasion_success` computes the boundary growthrate of one species with 
    different parameters
    
`coex_boundary` finds the boundaries for rel_fit for coexistence

`plt_coex_reg` plots the region of coexistence

`smooth` is a function to avoid numerical artifacts

`volume_coex_reg` computes the volume of the coexistence region
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D #implicitly used
from scipy.signal import convolve2d

import load_pigments as ld
lambs = ld.lambs
dlam = ld.dlam

def equilibrium(pigs, ratio, fit, plot = False):
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
    # absorption, sum(pig_i(lam)*alpha_i)
    absor = lambda ratio: np.einsum('pr,px->rx',[ratio, 1-ratio],
                            np.array([pig for pig in pigs]))
    # function to be integrated I_in*(1-e^-N*f*absor(lambda))
    iterator = lambda N_fit, ratio: 40/300*(1-np.exp(-
                        np.einsum('rf,rx->rfx',N_fit,absor(ratio))))
    #iteratively search for equilibrium, start at infinity
    equi = np.infty*np.ones([len(ratio), len(fit)])
    for i in range(25):
        # N*fit
        N_fit = np.einsum('rf,f->rf', equi,fit)
        #compute the function values
        y_simps = iterator(N_fit, ratio)
        #int(iterator, dlambda, 400,700)
        equi = simps(y_simps, dx = dlam)
    if np.amin(equi)<1:
        print("Warning, not all resident species have a equilibrium "+
              "density above 1")
        equi[equi<1] = np.nan
        
    if plot:
        fig = plt.figure(figsize = (7,7))
        X,Y = np.meshgrid(np.log(fit), ratio)
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(Y,X,equi)
        ax.set_xlabel('ratio of pig 1')
        ax.set_ylabel('Fitness (log)')
        ax.set_zlabel('Equilibrium density')
    return equi
    
def invasion_success(pigs, ratio, relfit, I_in = None, approx = False
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
        invade: array, invade.shape = (len(ratio), len(ratio), len(relfit))
            invade[i,j,k] is the boundary growhtrate of speces 2 with:
            resident has pig ratio ratio[i]
            invader has pig ratio ratio[j]
            relativefitness of species 2 is relfit[k]
    
    indexes in Einstein sumconvention:
        x-> x_simps
        r-> ratios of resident
        i-> ratios of invader
        f-> fitness
        p-> pigments"""
    if I_in is None:
        I_in = lambda lam: 40/300 #corresponds to constant light with lux = 40
    # vector of ratios for invader and resident
    r_inv = np.array([ratio, 1-ratio])
    r_res = np.array([ratio, 1-ratio])
    # sum(f*alpha_(p,i)*k_p(lambda)) summed over p
    numerator = np.einsum('px,pif->xif',pigs,
                          np.einsum('pi,f->pif',r_inv,relfit))
    # sum(alpha_(p,r)*k_p(lambda)) summed over p
    denominator = np.einsum('px,pr->xr',pigs,r_res)
    # (numerator/denominator -1)*I_in(lambda)
    rel_abs = (np.einsum('xif,xr->xirf',numerator,1/denominator)-1)*40/300

    if approx: # simplified version
        return simps(rel_abs, dx = dlam, axis = 0)
    f1 = avefit/np.sqrt(relfit)    #fitness of resident species
    equis = equilibrium(pigs, ratio,f1)#equilibrium of res
    # (N_equi*f1)*denominator
    expon = np.einsum('rf,xr->xrf',np.einsum('rf,f->rf', equis,f1),denominator)
    # rel_abs*(1-np.exp(-expon))
    y_simps = np.einsum('jkil,jil->jkil',rel_abs, 1-np.exp(-expon))
    invade = simps(y_simps, dx = dlam, axis = 0) #integrate
    return invade

def coex_boundary(pigs, ratio, fit, avefit = 1.38e8,sig= 0.0005):
    """Find the variables for which two species with `pigs` coexist
    
    Parameters:
        pigs: list of two pigments
            The pigments that compose the abs. spectrum for the two species
        ratio: array
            Ratio of pigment pigs[0] for each species
        fit: array
            relative fitness of the two species
        ave_fit: float, optional
            geometric mean of fitness of the two species
        sig:  float, optional
            sigma value for convolution
        
    Returns:
        convzmin: array
            convzmin.shape = (len(ratio), len(ratio)). Each entry in convzmin
            is the minimum for rel_fit that allows coexistence for these 
            settings of pigment ratios.
        convzmax: array
            similar to convzmin, but the maximum of rel_fit
     """
    #compute pigment distance
    pd = invasion_success(pigs, ratio,fit,avefit = avefit)
    coex_trip = np.where(pd>0) #case where invasion is possible
    nocoex_trip = np.where(pd<0) #no invasion possible
    
    #invert species 1 and 2 to switch invasion and resident
    nocoex_trip = [nocoex_trip[1], nocoex_trip[0], len(fit)-1-nocoex_trip[2]]
    # find all points that are only in coex_trip and not in nocoex_trip
    coex_set = set([tuple(x) for x in np.array(coex_trip).T])
    nocoex_set = set([tuple(x) for x in np.array(nocoex_trip).T])
    coex = np.array([x for x in coex_set-nocoex_set])

    #to save the maximum values for fitness, (len(fit)-1)/2 is for default value
    zval = (len(fit)-1)/2*np.ones([len(ratio), len(ratio),2])
    for i in range(len(ratio)):
        for j in range(len(ratio)):
            try: 
                #all points, that have the ratios [ratio[i], ratio[j]]
                r_ratio = coex[np.logical_and(coex[:,0]==i,coex[:,1]==j)][:,2]
                #save min and max of all these points
                zval[i,j,0] = np.amin(r_ratio)
                zval[i,j,1] = np.amax(r_ratio)
            except ValueError: #r_ratio is empty, no coexistence for these ratios
                pass
    #convert to ints to be able to call arrays
    zval = zval.astype(int)
    #convert index to actual fitness
    zmin = np.log(np.array([fit[i] for i in zval[:,:,0]]))
    zmax = np.log(np.array([fit[i] for i in zval[:,:,1]]))
    #smoothen min and max because of numerical error
    convzmin = smooth(zmin,sig)
    convzmax = smooth(zmax,sig)
    #convolution reduces maximum and minimum
    print("Reduction of maximum by convolution:"
              ,1-np.amax(zmax)/np.amax(convzmax))
    return convzmin, convzmax
    
def plt_coex_reg(pigs, grid_points = 51, vol = False):
    """plot the oexistence regions for the pigments pigs
    
    Plots the boundaries of the coexistence region in the 3-dim cube with the
    axis: ratio of pigment 1 for species 1, ratio of pigment 2 for species 2
    relative fitness
    
    parameters:
        pigs: list of two pigments
        grid_points: int, number of points in grid (-> accuracy)
        vol: bool, Optional
            If True, the relative volume of the coexistence region is returned
    """
    fit = 2**np.linspace(-1,1,grid_points) #uniform distributionin logspace
    ratio = np.linspace(0,1,grid_points) #ratio of pigments pigs[0]
    convzmin, convzmax = coex_boundary(pigs, ratio,fit)
    
    fig = plt.figure(figsize = (7,7))
    X,Y = np.meshgrid(ratio, ratio)
    ax = fig.gca(projection = '3d')
    
    
    ax.set_xlabel('ratio of pig 1 for species 1')
    ax.set_ylabel('ratio of pig 1 for species 2')
    ax.set_zlabel('relative fitness (log)')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    #where no coexistence is possible independent of rel_fit
    no_coex = convzmax.copy()
    no_coex[convzmax-convzmin>0.01] = np.nan

    #plot the grids
    ax.plot_wireframe(Y,X,convzmin, linewidth = 1,
                     color = 'green', rstride = 5, cstride = 5)
    ax.plot_wireframe(Y,X,convzmax, linewidth = 1
                , rstride = 5, cstride = 5, color = 'blue')
    ax.plot_wireframe(Y,X,no_coex, linewidth = 2
                , rstride = 5, cstride = 5, color = 'red')
    if vol:
        return volume_coex_reg(convzmin, convzmax)
    return fig
    
def smooth(mat,sig):
    """Snoothen `mat` with a gaussian kernel
    
    mat: 2-dim matrix taht needs to be smoothened, must be quadratic
    sig: shape of gaussian kernel
    
    out: mat, convolved with a gaussian kernel (e^(-|x|^2/sig**2)
    """
    if sig == 0: #no convolution
        return mat
    #compute the gaussian kernel
    points = np.linspace(0,1, mat.shape[0])
    x,y = np.meshgrid(points, points, sparse = True)
    kern = np.exp(-((x-0.5)**2+(y-0.5)**2)/(2*sig))
    # return smoothened/convolved array
    return convolve2d(mat, kern/kern.sum(), boundary = 'symm', mode = 'same')
    
def volume_coex_reg(convzmin, convzmax):
    """compute the relative volume of coexistence"""
    volsimp = simps(convzmax-convzmin,dx = 1/(convzmax.shape[0]-1))
    volsimp = simps(volsimp,dx = 1/(convzmax.shape[0]-1))
    # divide by volume of the cube
    return volsimp/(2*np.log(2))
    
import load_pigments as ld
fig = plt_coex_reg(ld.real[:2])
fig.savefig("Figure, 2species, coex volume.pdf")