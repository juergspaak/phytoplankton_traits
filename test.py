# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:46:20 2017

@author: spaakjue
"""
from scipy.integrate import ode

"""
N_timer = np.zeros([500,2])
N_timer[0] = N_start
itera = int(50)
h = 500.0/itera

for i in range(itera-1):
    N_timer[i+1] = N_timer[i]+h*growth(N_timer[i],1,resi)
"""   
    
def compare(N, resi, m = 15):
    simple = res_absorb_growth(N, 1,resi, m)
    comple = growth(N, 1, resi)
    return 1-simple/comple
    
print(compare(N_start, resi))