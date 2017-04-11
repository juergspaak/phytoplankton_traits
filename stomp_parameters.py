# -*- coding: utf-8 -*-
"""
Generates parameters for the stomp model
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint 

import help_functions_chesson as ches_funs
from help_functions_chesson import alpha

time = np.linspace(0,500,50)

resi = 0
N_start = np.array([10.0**5, 10.0**5])
N_start[resi] = 10.0**8
start = timer()
N_time = odeint(ches_funs.res_absorb_growth, N_start,time, args = (resi,))
print(timer()-start)

plt.plot(time,N_time)

plt.plot(time,ches_funs.absor_val['test_data'][resi],'^')
plt.figure()
plt.plot(time, 1-(N_time/ches_funs.absor_val['test_data'][resi]))