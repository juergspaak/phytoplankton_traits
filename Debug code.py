# -*- coding: utf-8 -*-
"""
These file has possibly no stable functions, it is only to try stuff
"""

def plot_I_out(N_res, resident):
    t = 0
    abs_fun = lambda lam: (N_res*k(lam))[resident] #only take the absorbing one
    I_out = lambda lam: I_in(t, lam)*np.exp(-abs_fun(lam)*zm)

    plotter(lambda lam: I_out(lam), 400,700)
    print(quad(lambda lam: I_out(lam),400,700)[0])
    
plot_I_out(10**9,1)
plot_I_out(10**9,0)

def alpha(n,resident, spe_int, t = 0):
    alpha = phi[spe_int]*(-zm)**n/math.factorial(n+1)
    alpha *= quad(lambda lam: k(lam)[spe_int]*k(lam)[resident]**n*I_in(t,lam)
            ,400,700)[0]
    if n == 0: alpha -=l[spe_int]
    return alpha

alphas = [np.array([[alpha(14-i,0,0),alpha(14-i,0,1)] for i in range(15)]),
          np.array([[alpha(14-i,1,0),alpha(14-i,1,1)] for i in range(15)])]


def nonexact_int_function(N_res,resident):
    print(np.polyval(abs_values[resident][:,resident][::-1],-N_res*zm))
    t = 0
    abs_fun = lambda lam: (N_res*k(lam))[resident] #only take the absorbing one
    fun = lambda lam: I_in(t, lam)*(1-np.exp(-abs_fun(lam)*zm))
    print(quad(fun, 400,700)[0])
    
nonexact_int_function(10**9,1)