"""
@author: Jurg W. Spaak
Computes the boundary growth for both species with analytical solution
and continuous time
Assumes that I_out = 0
"""
import numpy as np
from scipy.integrate import simps

import analytical_communities as com

def cumsimps(y, x=None, dx=1.0, axis=-1, initial = None):
    """Cumulatively integrate y(x) using the composite simpson rule.
    
    Parameters:	
        y : array_like
            Values to integrate.
        x : array_like, optional
            The coordinate to integrate along. If None (default), use spacing 
            dx between consecutive elements in y.
        dx : int, optional
            Spacing between elements of y. Only used if x is None.
        axis : int, optional
            Specifies the axis to cumulate. Default is -1 (last axis).
        initial : scalar, optional
            If given, uses this value as the first value in the returned 
            result. Typically this value should be 0. Default is None, which
            means no value at x[0] is returned and res has one element less 
            than y along the axis of integration.
        
        Returns:	
            res : ndarray
                The result of cumulative integration of y along axis. 
                If initial is None, the shape is such that the axis of 
                integration has one less value than y. If initial is given, the
                shape is equal to that of y."""
    if y.shape[axis]%2==0: # check input, simpson need odd number of intervals
        raise ValueError("y must have an odd number of elements in axis")
    if x is None:#create x, multiply dx by 2, as the interval contains 3 points
        x = 2*dx*np.ones(int((y.shape[axis]-1)/2))
        x.shape = (-1,)+(y.ndim-axis%y.ndim-1)*(1,)
    # get the function values
    end = y.shape[axis]-1
    fa = np.take(y,range(0,end,2), axis = axis) # f(a), start of each interval
    fab = np.take(y,range(1,end+1,2), axis = axis) # f((a+b)/2), midpoint
    fb = np.take(y,range(2,end+1,2), axis = axis) # f(b), endpoint
    Sf = x/6*(fa+4*fab+fb) # actual integration, see simpson rule
    if initial is None: # no start value given
        out = np.cumsum(Sf, axis = axis)
    else: # add starting value
        shape = list(Sf.shape)
        shape[axis] +=1
        out = np.full(shape,initial,dtype = float)
        idx = [slice(None)]*y.ndim
        idx[axis] = slice(1,None)
        out[tuple(idx)] = np.cumsum(Sf,axis = axis)+initial
    return out
    
def resident_density(species, I,t):
    k,H,p,l = species # species parameters
    # equilibrium densities for incoming light
    W_r_star = com.equilibrium(species, I(t)[::2], "full")
    t2 = np.append(t[1:],t+t[-1]) #compute over two periods to find equilibria
    t2.shape = -1,1,1
    int_fun = np.exp(l*t2)/(H+I(t2))*I.dt(t2)
    dt = (t[-1]-t[0])/t.size
    W_r_diff = p/(k*l)*cumsimps(int_fun,dx = dt, axis = 0, initial = 0)\
                    *np.exp(-l*t2[::2])
    W_r_t = W_r_star-W_r_diff[(t.size-1)//2:]
    """plt.plot(t,I(t))
    plt.figure()
    plt.plot(t[::2], W_r_star[:,0,0])
    plt.plot(t[::2], W_r_t[:,0,0])"""
    return W_r_t, W_r_star
    
def continuous_r_i(species, I,t):
    # computes the boundary growth rates for invading species
    W_r_t, W_r_star = resident_density(species, I, t)
    i,r = [[0,1],[1,0]]
    k,l = species[[0,-1]]
    dt = (t[-1]-t[0])/t.size
    simple_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_star[:,r])-1,
                       dx = dt,axis = 0)*l[i]/t[-1]
    exact_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_t[:,r])-1,
                       dx = dt,axis = 0)*l[i]/t[-1]
    return simple_r_i, exact_r_i
    
    

    
T = 1*2*np.pi
t,dt = np.linspace(0,5*T,1001, retstep = True)
size = 80
I = lambda t: size*np.sin(t/T*2*np.pi)+125
I.dt = lambda t:  size*np.cos(t/T*2*np.pi)*2*np.pi/T
species = com.gen_species(1000)

simple_r_i, exact_r_i = continuous_r_i(species, I, t)
plit(*simple_r_i)
plit(*exact_r_i)
plot_percentiles(np.abs((exact_r_i-simple_r_i)/exact_r_i), y_max = 1)
plt.show()
print(np.sum(np.sum(simple_r_i>0, axis = 0)==2))
print(np.sum(np.sum(exact_r_i>0, axis = 0)==2))

    