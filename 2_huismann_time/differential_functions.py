"""
@author: J.W.Spaak, jurg.spaak@unamur.be

Contains functions to solve differential equations
"""
import numpy as np

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
    
def own_ode(f,y0,t, *args, steps = 10, s = 2):
    """uses adam bashforth method to solve an ODE
    
    Parameters:
        func : callable(y, t0, ...)
            Computes the derivative of y at t0.
        y0 : array
            Initial condition on y (can be a vector).
        t : array
            A sequence of time points for which to solve for y. 
            The initial value point should be the first element of this sequence.
        args : tuple, optional
            Extra arguments to pass to function.
            
    Returns:
        y : array, shape (steps, len(y0))
            Array containing the value of y for each desired time in t,
            with the initial value y0 in the first row."""
    # coefficients for method:
    coefs = [[1], # s = 1
             [-1/2,3/2],  # s = 2
             [5/12, -4/3, 23/12], # s = 3
             [-3/8, 37/24, -59/24, 55/24], # s = 4
             [251/720, -637/360, 109/30, -1387/360, 1901/720]] # s = 5
    coefs = np.array(coefs[s-1]) #choose the coefficients
    ts, h = np.linspace(*t, steps, retstep = True) # timesteps
    # to save the solution and the function values at these points
    sol = np.empty((steps,)+ y0.shape)
    dy = np.empty((steps,)+ y0.shape)
    #set starting conditions
    sol[0] = y0
    dy[0] = f(y0, ts[0], *args)
    # number of points until iteration can start is s
    for i in range(s-1): #use Euler with double precision
        sol_help = sol[i]+h/2*f(sol[i],ts[i], *args)
        sol[i+1] = sol_help+h/2*f(sol_help,ts[i]+h/2)
        dy[i+1] = f(sol[i+1], ts[i+1], *args)
        
    #actual Adams-Method
    for i in range(steps-s):
        sol[i+s] = sol[i+s-1]+h*np.einsum('i,i...->...', coefs, dy[i:i+s])
        dy[i+s] = f(sol[i+s], ts[i+s],*args)
    return sol