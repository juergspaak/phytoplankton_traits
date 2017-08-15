"""
this programm assumes that the growth of the invader is simply:
    (k_i*W_i(T+1)/(k_r*W_r(T+1)-1)*l_i*P #first term of full solution
    
The boundary growth rate of both species is positive if
E[k_i*W_i(T+1)/(k_r*W_r(T+1))]>1
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps


def check_IR(Im=50, IM=200, samples=100):
    #distribution of lights for integration
    I_eff, dI = np.linspace(Im,IM,51, retstep = True) 
    I_eff.shape = (-1,1,1)
    #distribution of halfsaturation for both species
    H1, H2 = np.random.uniform(1,10000,(2,samples))
    H2.shape = (-1,1)
    # k_i*l_i/p_i*W_i(T+1)/(k_r*l_r/p_r*W_r(T+1))
    equi_rel = np.log(1+I_eff/H1)/np.log(1+I_eff/H2)
    #integrate for first species
    ave = simps(equi_rel, dx = dI, axis = 0)/(IM-Im)
    #integrate for second species
    ave_inv = simps(1/equi_rel, dx = dI, axis = 0)/(IM-Im)
    f_min = 1/ave # f = k_i*l_i/p_i/(k_r*l_r/p_r)
    f_max = ave_inv
    # return maximal value range for coefficients    
    return np.amax((f_max-f_min)/f_max)
    
"""    
To test the parameters, find maximal range for coefs to coexist
Im = np.random.uniform(10,1000, 100)  
IM = Im*np.random.uniform(1,3,100)
max_diff = 0
for i in range(100):
    max_diff = max(check_IR(Im[i], IM[i]),max_diff)"""
    

def reduce_2_var(I_M_max = 1000, H_max = 1e5):
    """computes the integral:
        \int_0^(I_M) log(1+I/H)/log(1+I) dI
    and plots the result in a color map, axes are logarithmic"""
    
    acc = 100
    dist,dx = np.linspace(-1,1,acc, retstep = True)
    H = H_max**dist
    I_M = I_M_max**dist
    
    x_simps = np.linspace(0,1,101)
    lights = I_M.reshape((-1,1))*x_simps
    y_simps = np.log(1+lights/H[:,None,None])/np.log(1+lights)
    y_simps[np.isnan(y_simps)] = 1
    integral = simps(y_simps, dx = dx, axis = -1)*I_M
    plt.imshow(np.log(integral), cmap = 'hot')
    plt.colorbar()
    return integral
    
save = reduce_2_var()