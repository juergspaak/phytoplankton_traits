"""@author: J.W.Spaak
contains functions that solve the ode for many communities"""

import numpy as np

def own_ode(f,y0,t,args = (), steps = 10, s = 2 ):
    """uses adam bashforth method to solve an ODE
    
    Parameters:
        func : callable(y, t0, ...)
            Computes the derivative of y at t0.
        y0 : array
            Initial condition on y (can be a vector).
        t : array
            start and endpoint of interval to obtain sol
        steps: integer
            number of steps to compute sol
        args : tuple, optional
            Extra arguments to pass to function.
            
    Returns:
        y : array, shape: (steps,) + y0.shape 
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
    sol[0] = y0 # solutions
    dy[0] = f(y0, ts[0], *args) # save differentials at timepoints
    # explicit midpoint rule
    midpoint = lambda yn,tn: yn+h*f(yn+h/2*f(yn,tn, *args),tn+h/2,*args)
    # number of points until iteration can start is s
    for i in range(s-1): # use Euler with double precision
        sol[i+1] = midpoint(sol[i], ts[i])
        dy[i+1] = f(sol[i+1], ts[i+1], *args)
        
    #actual Adams-Method
    for i in range(steps-s):
        sol[i+s] = sol[i+s-1]+h*np.einsum('i,i...->...', coefs, dy[i:i+s])
        dy[i+s] = f(sol[i+s], ts[i+s], *args)
    return sol
    
if __name__ == "__main__":
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    
    factor = np.arange(1,4)
    fun = lambda x,t:x*factor
    start = np.ones(3)
    times = np.linspace(0,1,10)
    sol_exact = odeint(fun,start, times)
    sol_own = np.empty((5,10,3))
    col = ["r", "g","c","p","y"]
    for s in range(5):
        sol_own[s] = own_ode(fun, start, times[[0,-1]],s = s+1)
        plt.plot(sol_own[s],col[s], label = s+1)
    plt.plot(sol_exact,'bo', label = "exact")
    plt.legend(loc = "upper left")
