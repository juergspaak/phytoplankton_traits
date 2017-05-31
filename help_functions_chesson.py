
"""
Created on Fri Apr  7 09:14:35 2017

@author: spaakjue
Contains help functions for stomp model
"""
import math
import numpy as np

from scipy.integrate import quad, odeint
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
from numpy.random import uniform as uni


"""loading/defining the variables"""
absor_val = np.load("npz,stomp_values.npz") #loads the absorption curves of the cyanos
alphas_global = absor_val['alphas']

k_red = interp1d(absor_val['x_red'], 10**-9*absor_val['y_red'], 'cubic')
k_green = interp1d(absor_val['x_green'], 10**-9*absor_val['y_green'], 'cubic')
k = lambda lam: np.array([k_green(lam), k_red(lam)])

l = np.array([0.014,0.014])  #specific loss rate [h^-1]

phi = 10**6*np.array([2.2,1.6])   # photosynthetic efficiency [fl*mumol^-1]
zm = 7.7          #total depth [m]
N = np.array([1,1]) # density [fl*cm^-3]
I_in_prev = lambda t,l: 1


#####################
def alpha(n,resident, spe_int, t = 0):
    """computes the taylor expansion for the differential equation
    
    m = 29
    alphas = np.array([[[alpha(m-i,0,0),alpha(m-i,0,1)] for i in range(m+1)],
    [[alpha(m-i,1,0),alpha(m-i,1,1)] for i in range(m+1)]])
    contains the values phi*(zm)^n/(n+1)!*integrate(k_spec*k_res^n*I_in dlambda)
    alphas[res][n,spec] contains those values"""
    
    alpha_fac = phi[spe_int]*(-zm)**n/math.factorial(n+1)
    #splitting the integral into these parts, because they have really
    #different order of magnitudes, quad seems to be messing these things up          
    alpha1 = quad(lambda lam: k(lam)[spe_int]*k(lam)[resident]**n
            ,400,500)[0]
    alpha2 = quad(lambda lam: k(lam)[spe_int]*k(lam)[resident]**n
            ,500,600)[0]
    alpha3 = quad(lambda lam: k(lam)[spe_int]*k(lam)[resident]**n
            ,600,700)[0]   
    #fun = lambda lam: k(lam)[spe_int]*k(lam)[resident]**n
    #plt.semilogy(np.linspace(400,700,100),fun(np.linspace(400,700,100)))
    #lam = 500
    #print(k(lam)[spe_int]*k(lam)[resident]**n)
    return alpha_fac*(alpha1+alpha2+alpha3)

class LuxIn:
    """class that maintains the amount of incoming light"""
    def __init__(self, period):
        self.period = period # period lenght of environment
        self.record_lux = [uni(20,60)] #current incomming light]
    
    def time(self,t):
        """if t reaches the end of the period, the incoming light is changed"""
        if (t//self.period)>(len(self.record_lux)-1):
            new_values = uni(20,60,t//self.period-len(self.record_lux)+1)
            self.record_lux.extend(new_values)
        return self.record_lux[int(t//self.period)]

    
def alphas_time(t, lux_in):
    alphas = alphas_global*lux_in.time(t)/300 #alphas were computed with normalized light intensity
    alphas[0][-1,:] -= l #dilution rate added to be flexible with light intensity
    alphas[1][-1,:] -= l 
    return alphas


def outcoming_light(N,t, absor = 'both'):
    """computes the outcoming light, i.e. flux"""
    if absor == 'both':
        abs_fun = lambda lam: sum(N*k(lam)) #take the sum
    else: 
        abs_fun = lambda lam: (N*k(lam))[absor] #only take the absorbing one
    I_out = lambda lam: I_in(t, lam)*np.exp(-abs_fun(lam)*zm)
    return quad(I_out, 400,700)[0]    

    
################## function not using taylor expansion (most exact)
def growth(N, t, absor, lux = 40):
    """computes the growth of both species
    
    absor can be 'both', i.e. both species absorb or an integre (0 or 1)
    indicating that only this species absorbs light (the other is at 0 density)
    """
    if absor == 'both':
        abs_fun = lambda lam: sum(N*k(lam)) #take the sum
    else: 
        abs_fun = lambda lam: (N*k(lam))[absor] #only take the absorbing one
    #plotter(abs_fun,400,700)
    I_in = lambda lam: lux/300
    integrand = lambda lam, col: I_in(lam)*k(lam)[col]/abs_fun(lam)*\
                            (1-math.exp(-abs_fun(lam)*zm))
    gamma0 = quad(lambda lam: integrand(lam,0), 400,700)[0]
    gamma1 = quad(lambda lam: integrand(lam,1), 400,700)[0]
    gamma = np.array([gamma0,gamma1])
    return (phi/zm*gamma-l)*N
    

###################### functions making use of Taylor expansion
times = len(alphas_global[0][:,0])
def res_absorb_growth(N,t,resident,lux_in, m = 15):
    """computes the growthrate when only one species is absorbing
    
    resident is the integer of the resident species (0 or 1)    
    if N is an array (i.e contains the values for both species), it returns
    the growthrate for both, otherwise N should be the density of the resident
    m is the order of the taylor approximation, max = len(alphas[0][:,0])=29,
    should be uneven
    
    This is done by a tylor approximation up to 16 terms. The code in the if
    statement is equivalent N*np.polyval(alphas[resident], N[resident]),
    but about 10-20% faster"""
    alphas = alphas_time(t, lux_in)
    if m%2==1 and m<30:
        if type(N) == np.ndarray and len(N)==2:
            return N*np.polyval(alphas[resident][times-m:,:], N[resident])
        else:
            return N*np.polyval(alphas[resident][times-m:,resident],N)
    else: print("choose m uneven for stability reasons and m<30 is necessray")

##################### functions that use anaylitcal integration
def analytical_integral(coefs):
    """analyticaly solves the differential equation dN/dt = 1/sum(N^i*coefs[n-i])
    returns the result in a function fun(N) = t+c, which can be solved
    numerically for solutions. returns the function fun, as well as d/dN fun(N)
    
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
    
def N_time_fun(N_start,spe_int,lux_in,t = 0):
    """returns a callable, that has the densities of N over time
    
    N_start is the starting density of N, coefs are the taylor approximation
    for the stomp differential equation whith only one absorber
    
    warning: is numerically unstable near the equilibrium"""
    alphas = alphas_time(t, lux_in)
    coefs = np.append(alphas[spe_int][:,spe_int],0) #multipling with species
    N_fun, N_fun_prime = analytical_integral(coefs)     ##densities
    solver_fun = lambda N,t: N_fun(N)-N_fun(N_start)-t
    maximu = stomp_equi(spe_int)
    return lambda t: brentq(solver_fun,10**8,maximu,args = (t,))

    

def invader_growth(N_start, resi, end_time,lux_in, N_time = None,
                   t =0, accuracy = 100):
    """returns invader density at end_time
    
    could be updated to return a fucntion/array"""
    alphas = alphas_time(t,lux_in)
    inv = int(not resi)
    if N_time is None:
        time = np.linspace(0,end_time,accuracy)
        N_time = odeint(res_absorb_growth, N_start[resi],time, args = (resi,))
    else:
        time = np.linspace(0,end_time, len(N_time))
    #plt.plot(time,N_time)
    N_times = N_time[:,0]**exponent
    #resident_fun = interp1d(time, N_time, 'cubic')
    integrates = np.trapz(N_times, time)
    return N_start[inv]*np.exp(sum(alphas[resi][:,inv]*integrates))
  
    
def stomp_equi(spe_int,t,lux_in,start = False, acc = 0.01):
    """returns the equilibrium density of species spe_int in monoculture
    
    if start is a float, then it returns aswell the time to reach 
    a range of acc within the equilibrium, with starting density start"""
    alphas = alphas_time(t,lux_in)
    coefs = alphas[spe_int][:,spe_int]
    roots = np.roots(coefs)
    equi = np.real(roots[-1])
    if not start:
        return equi
    N_fun, N_fun_prime = analytical_integral(np.append(coefs,0))
    return equi, N_fun(equi*(1-acc))-N_fun(start)