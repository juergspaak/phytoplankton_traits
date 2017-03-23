# -*- coding: utf-8 -*-
"""
Generates parameters for the stomp model
"""
from scipy.interpolate import interp1d
import math
from scipy.integrate import quad, odeint
import numpy as np
import matplotlib.pyplot as plt

absor_val = np.load("npz,absorption.npz") #loads the absorption curves of the cyanos


k_red = interp1d(absor_val['x_red'], 10**-9*absor_val['y_red'], 'cubic')
k_green = interp1d(absor_val['x_green'], 10**-9*absor_val['y_green'], 'cubic')
k = lambda lam: np.array([k_green(lam), k_red(lam)])

l = np.array([0.014,0.014])  #specific loss rate [h^-1]

phi = 10**6*np.array([1.6,1.6])   # photosynthetic efficiency [fl*mumol^-1]
zm = 7.7          #total depth [m]
N = np.array([1,1]) # density [fl*cm^-3]
I_in_prev = lambda t,l: 1
int_I_in = 40  # light over entire spectrum [mumol ph*m^-2*s^-1]
I_in = lambda t,l: I_in_prev(t,l)*int_I_in/300

def growth(N, t, absor = 'both' ):
    if absor == 'both':
        abs_fun = lambda lam: sum(N*k(lam)) #take the sum
    else: 
        abs_fun = lambda lam: (N*k(lam))[absor] #only take the absorbing one
    #plotter(abs_fun,400,700)
    integrand = lambda lam, col: I_in(t,lam)*k(lam)[col]/abs_fun(lam)*\
                            (1-math.exp(-abs_fun(lam)*zm))
    gamma0 = quad(lambda lam: integrand(lam,0), 400,700)[0]
    gamma1 = quad(lambda lam: integrand(lam,1), 400,700)[0]
    gamma = np.array([gamma0,gamma1])
    return (phi/zm*gamma-l)*N

def outcoming_light(N,t, absor = 'both'):
    """computes the outcoming light"""
    if absor == 'both':
        abs_fun = lambda lam: sum(N*k(lam)) #take the sum
    else: 
        abs_fun = lambda lam: (N*k(lam))[absor] #only take the absorbing one
    I_out = lambda lam: I_in(t, lam)*np.exp(-abs_fun(lam)*zm)
    return quad(I_out, 400,700)[0]



    
    
def intens(n, resident, invader = None, t = 0):
    if invader == None:
        invader = resident
    abs_n = lambda lam: k(lam)[invader]*k(lam)[resident]**n*I_in(t,lam)
    
    return quad(abs_n,400,700)[0]
    
fun = lambda n: phi[1]*(-N_time[-1]*zm)**n/(math.factorial(n+1))
sumor = lambda n,k,l: sum([fun(n-1-i)[1]*intens(n-1-i,k,l) for i in range(n)])

abs_values = [np.array([[intens(i,0,0),intens(i,0,1)] for i in range(15)]),
             np.array([[intens(i,1,0),intens(i,1,1)] for i in range(15)])]

exponent = [[i] for i in range(15)]
divisor = np.array([[math.factorial(i+1)] for i in range(15)])

def one_abs_growth(N, t, resident):
    """computes the growth rate, when only one species is absorbing light"""
    N_values = (-N*zm)**exponent/divisor
    return N*(phi*sum(N_values*abs_values[resident])-l)
    

time = np.linspace(0,500,50)
start = timer()
N_time = odeint(one_abs_growth, np.array([10**8,0]),time, args = (0,))
plt.plot(time,N_time)
print(timer()-start)
plt.plot(time,save,'^')