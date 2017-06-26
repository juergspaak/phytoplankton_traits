# -*- coding: utf-8 -*-
"""
Generates parameters for the stomp model
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint


import help_functions_chesson as ches_funs
from help_functions_chesson import alpha

time = np.linspace(1e-5,500,50) #start not at zero to prevent resetting of lux

lux_in = ches_funs.LuxIn(1000)
lux_in.record_lux[0] = 40

resi = 1
N_start = np.array([10.0**5, 10.0**5])
N_start[resi] = 10.0**8
start = timer()
N_time = odeint(ches_funs.res_absorb_growth, N_start,time, args = (resi,lux_in))
print(timer()-start)

plt.plot(time,N_time)

if False and time[-1] == 500 and len(time) == 50:
    plt.plot(time,ches_funs.absor_val['test_data'][resi],'^')
    plt.figure()
    plt.plot(time, 1-(N_time/ches_funs.absor_val['test_data'][resi]))
    
#N_check = odeint(ches_funs.growth,N_start, time, args = (resi,))
