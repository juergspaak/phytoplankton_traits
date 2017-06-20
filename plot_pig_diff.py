import numpy as np
import help_functions_chesson as ches
import different_pigments as dpig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
from timeit import default_timer as timer
from scipy.integrate import simps

#parameters for the run
npoints = 101
fit = 2**np.linspace(-1,1,npoints) #uniform distributionin logspace
ratio = np.linspace(0,1,npoints)
avefit = 1.38e8
abs_fit = avefit/np.sqrt(fit)

def plt_equi():
    #plot equilibrium data
    equis = dpig.equilibrium([ches.k_green, ches.k_red], ratio,abs_fit)
    fig = plt.figure(figsize = (7,7))
    X,Y = np.meshgrid(np.log(abs_fit), ratio)
    ax = fig.gca(projection = '3d')
    ax.plot_wireframe(Y,X,equis)
    ax.set_xlabel('ratio of pig 1')
    ax.set_ylabel('Fitness (log)')
    ax.set_zlabel('Equilibrium density')
    
def smooth(mat,sig=1):
    """Make a gaussian kernel, e^(-|x|^2/sig**2)"""
    if sig == 0: #no convolution
        return mat
    points = np.linspace(0,1, mat.shape[0])
    x,y = np.meshgrid(points, points, sparse = True)
    kern = np.exp(-((x-0.5)**2+(y-0.5)**2)/(2*sig))
    return convolve2d(mat, kern/kern.sum(), boundary = 'symm', mode = 'same')

def pig_diff_points(avefit = avefit):

    #compute pigment distance
    pd = dpig.pigments_distance([ches.k_green, ches.k_red], ratio,fit
                                ,approx = False,avefit = avefit)
    coex_trip = np.where(pd>0) #case where invasion is possible
    nocoex_trip = np.where(pd<0) #no coexistence possible
    
    #invert species 1 and 2 to switch invasion and resident
    nocoex_trip = [nocoex_trip[1], nocoex_trip[0], len(fit)-1-nocoex_trip[2]]
    # find all points that are only in coex_trip and not in nocoex_trip
    coex_set = set([tuple(x) for x in np.array(coex_trip).T])
    nocoex_set = set([tuple(x) for x in np.array(nocoex_trip).T])
    coex = np.array([x for x in coex_set-nocoex_set]).T
                    
    # convert to arrays for plotting                
    xs = np.array([ratio[i] for i in coex[0]])
    ys = np.array([ratio[i] for i in coex[1]])
    zs = np.array([fit[i] for i in coex[2]])
    return xs,ys,zs,coex
    
def plt_all_points(xs,ys,zs):
    #plot all points
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111,projection= '3d')
    ax.scatter(ys,xs,np.log(zs),s = 1, color = 'blue')
    ax.set_xlabel('ratio invader')
    ax.set_ylabel('ratio resident')
    ax.set_zlabel('relative fitness (log)')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

def min_max_pig_diff(xs,ys,zs,coex, sig= 0.0005):
    #to save the maximum values for fitness
    zval = (len(fit)-1)/2*np.ones([len(ratio), len(ratio),2])
    no_coex = np.nan*np.ones([len(ratio), len(ratio)])
    co = coex.T
    for i in range(len(ratio)):
        for j in range(len(ratio)):
            try: 
                zval[i,j,0] = np.amin(co[np.logical_and(co[:,0]==i, 
                                      co[:,1]==j)][:,2])
                zval[i,j,1] = np.amax(co[np.logical_and(co[:,0]==i, 
                                      co[:,1]==j)][:,2])
            except ValueError:
                no_coex[i,j] = 0
    
    zval = zval.astype(int)
    zmin = np.log(np.array([fit[i] for i in zval[:,:,0]]))
    convzmin = smooth(zmin,sig)
    zmax = np.log(np.array([fit[i] for i in zval[:,:,1]]))
    convzmax = smooth(zmax,sig)
    print(np.amax(zmax), np.amax(convzmax))
    return convzmin, convzmax

def plt_pig_diff(convzmin, convzmax):
    fig = plt.figure(figsize = (7,7))
    X,Y = np.meshgrid(ratio, ratio)
    ax = fig.gca(projection = '3d')
    
    
    ax.set_xlabel('ratio invader')
    ax.set_ylabel('ratio resident')
    ax.set_zlabel('relative fitness (log)')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    eq = convzmax.copy()
    eq[convzmax-convzmin>0.01] = np.nan
    
    
    ax.plot_wireframe(X,Y,convzmin, linewidth = 1,
                     color = 'green', rstride = 5, cstride = 5)
    ax.plot_wireframe(X,Y,convzmax, linewidth = 1
                , rstride = 5, cstride = 5, color = 'blue')
    ax.plot_wireframe(X,Y,eq, linewidth = 3
                , rstride = 5, cstride = 5, color = 'red')

def volume_pig_diff(zmin,zmax):
    volsimp = simps(zmax-zmin,dx = 1/(zmax.shape[0]-1))
    volsimp = simps(volsimp,dx = 1/(zmax.shape[0]-1))
    return volsimp
start = timer()
xs,ys,zs,coex = pig_diff_points()
print(timer()-start, " compute points")
zmin, zmax = min_max_pig_diff(xs,ys,zs,coex)
print(timer()-start, "find min, max") 
plt_pig_diff(zmin, zmax)
print(timer()-start, "plotting")
print(volume_pig_diff(zmin, zmax))