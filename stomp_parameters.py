# -*- coding: utf-8 -*-
"""
Generates parameters for the stomp model
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import odeint

"""loading/defining the variables"""
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

import help_functions_chesson as ches_funs




time = np.linspace(0,500,50)

resi = 0
N_start = np.array([10.0**5, 10.0**5])
N_start[resi] = 10.0**8
start = timer()
N_time = odeint(ches_funs.res_absorb_growth, N_start,time, args = (resi,))
print(timer()-start)
plt.plot(time,N_time)

plt.plot(time,absor_val['test_data'][resi],'^')
plt.figure()
plt.plot(time, 1-(N_time/absor_val['test_data'][resi]))