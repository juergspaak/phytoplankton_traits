"""
@author: Jurg W. Spaak

Find the maximal boundary growth rates for both species
assumes slow fluctuations, s.t. resident is always at equilibrium density"""

import numpy as np
import matplotlib.pyplot as plt




a_max = 1.5
b_min = 0.7
p = np.linspace(0,1,50)
q = 1-p
a_sqrt = p
b_sqrt = q*b_min-q/b_min
c_sqrt = -p

a = (-b_sqrt+np.sqrt(b_sqrt**2-4*a_sqrt*c_sqrt))/(2*a_sqrt)
a[a>a_max] = a_max



p_opt = (1/b_min-b_min)/(a_max-1/a_max+1/b_min-b_min)
opt_computed = p_opt*a_max+(1-p_opt)*b_min
opt_analytical = (a_max+b_min)/(1+a_max*b_min)
print(opt_computed, opt_analytical)


def n_numbers(n,itera):
    a_i = np.random.uniform(b_min,a_max,(n,itera))
    p_i = np.random.uniform(0,1,(n,itera))
    p_i = p_i/np.sum(p_i,axis = 0)
    E_x = np.sum(a_i*p_i,axis = 0)
    E_1_x = np.sum(p_i/a_i,axis = 0)
    fun_eff = np.amin(np.array([E_x,E_1_x]),0)
    i = np.argmax(fun_eff)
    print(np.amax(np.amin(np.array([E_x,E_1_x]),0)))
    print(a_i[:,i],p_i[:,i] )

    
    
n_numbers(10,10000000)
print([a_max,b_min], [p_opt, 1-p_opt])
