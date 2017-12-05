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
    
def resident_density(species, I,period, acc = 1001):
    # check input
    
    if np.exp(period*np.amax(species[-1]))==np.inf:
        raise ValueError("Too long `period` in `resident_density`."+
        "The product l*period must be smaller than 700 to avoid overflow.")
    k,H,p,l = species # species parameters
    # time for numerical integration
    t,dt = np.linspace(0,period,acc,retstep = True) 
    # equilibrium densities for incoming light
    W_r_star = com.equilibrium(species, I(t)[::2], "full")
    t.shape = -1,1,1
    int_fun = np.exp(l*t)/(H+I(t))*I.dt(t)
    # int_0^t e^(l*(s-t))/(H+I(s))*I.dt(s) ds, for t<period
    W_r_diff = p/(k*l)*cumsimps(int_fun,dx = dt, axis = 0, initial = 0)\
                    *np.exp(-l*t[::2])
    
    # Finding W_r_diff(t=0), using W_r_diff(0)=W_r_diff(T)
    W_r_diff_0 = W_r_diff[-1]/(1-np.exp(-l*period))           
    # adding starting condition
    W_r_diff = W_r_diff + W_r_diff_0*np.exp(-l*t[::2])
    # computing real W_r_t
    W_r_t = W_r_star-W_r_diff
    return W_r_t, W_r_star,dt
    
def continuous_r_i(species, I,period, acc = 1001):
    # computes the boundary growth rates for invading species
    W_r_t, W_r_star,dt = resident_density(species, I, period,acc)
    i,r = [[0,1],[1,0]]
    k,l = species[[0,-1]]
    simple_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_star[:,r])-1,
                       dx = dt,axis = 0)*l[i]
    exact_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_t[:,r])-1,
                       dx = dt,axis = 0)*l[i]
    return simple_r_i, exact_r_i