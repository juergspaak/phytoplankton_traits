# -*- coding: utf-8 -*-
"""
Generates parameters for the stomp model
"""
from scipy.interpolate import interp1d
import math
from scipy.integrate import quad, odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

absor_val = np.load("npz,stomp_values.npz") #loads the absorption curves of the cyanos
alphas = absor_val['alphas']

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
alphas = alphas*int_I_in/300 #alphas were computed with normalized light intensity
alphas[0][-1,:] -= l
alphas[1][-1,:] -= l

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
    
def alpha(n,resident, spe_int, t = 0):
    alpha = phi[spe_int]*(-zm)**n/math.factorial(n+1)
    alpha *= quad(lambda lam: k(lam)[spe_int]*k(lam)[resident]**n
            ,400,700)[0]
    return alpha

#alphas = np.array([[[alpha(15-i,0,0),alpha(15-i,0,1)] for i in range(16)],
#         [[alpha(15-i,1,0),alpha(15-i,1,1)] for i in range(16)]])
#contains the values phi*(zm)^n/(n+1)!*integrate(k_spec*k_res^n*I_in dlambda)
#alphas[res][n,spec] contains those values


def analytical_integral(coefs):
    """analyticaly solves the differential equation dN/dt = 1/sum(N^i*coefs[n-i])
    returns the result in a function fun(N) = t+c, which can be solved
    numerically for solutions. returns the function fun, as well ad d/dN fun(N)
    
    this is done by a partial fraction decomposiiton"""
    roots = np.roots(coefs)
    n = len(roots)
    mat = np.array([np.poly(np.delete(roots,i)) for i in range(n)])
    mat = np.matrix.transpose(mat)
    num_poly = np.zeros(len(roots)) 
    num_poly[-1]=1 #b are the coefs of the numerator polynom
    beta = np.linalg.solve(mat, num_poly)/coefs[0] #numerators of the partial fractions
    fun_prime = lambda N: np.array([np.real(sum(beta/(N-roots)))]) #fractial decomposition
    fun = lambda N: np.real(sum(beta*np.log(N-roots)))
    return fun, fun_prime
    
def N_time_fun(N_start,spe_int):
    """returns a callable, that has the densities of N over time
    
    N_start is the starting density of N, coefs are the taylor approximation
    for the stomp differential equation whith only one absorber
    
    warning: is numerically unstable near the equilibrium"""
    coefs = np.append(alphas[spe_int][:,spe_int],0) #multipling with species
    N_fun, N_fun_prime = analytical_integral(coefs)     ##densities
    solver_fun = lambda N,t: N_fun(N)-N_fun(N_start)-t
    return lambda t: fsolve(solver_fun,N_start,args = (t,),
                    fprime = lambda N,t: N_fun_prime(N))
    
def stomp_equi(spe_int,start = False, acc = 0.01):
    """returns the equilibrium density of species spe_int in monoculture
    
    if start is a float, then it returns aswell the time to reach 
    a range of acc within the equilibrium, with starting density start"""
    coefs = alphas[spe_int][:,spe_int]
    roots = np.roots(coefs)
    equi = np.real(roots[-1])
    if not start:
        return equi
    N_fun, N_fun_prime = analytical_integral(np.append(coefs,0))
    return equi, N_fun(equi*(1-acc))-N_fun(start)
    


times = len(alphas[0][:,0])
exponent = np.array([[times-1-i] for i in range(times)])

def res_absorb_growth(N,t,resident, precision = 0):
    """computes the growthrate when only one species is absorbing
    
    This is done by a tylor approximation up to 15 terms. The code is
    equivalent N*np.polyval(alphas[resident], N[resident]), but about
    10-20% faster"""
    N_values = N[resident]**exponent[precision:]
    return N*(sum(N_values*alphas[resident][precision:,:]))

time = np.linspace(0,500,50)

resi = 0
N_start = np.array([10.0**5, 10.0**5])
N_start[resi] = 10.0**8
start = timer()
N_time = odeint(res_absorb_growth, N_start,time, args = (resi,))
print(timer()-start)
plt.plot(time,N_time)

plt.plot(time,absor_val['test_data'][resi],'^')
plt.figure()
plt.plot(time, 1-(N_time/absor_val['test_data'][resi]))