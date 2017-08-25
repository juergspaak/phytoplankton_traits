"""
This file is to find the conditions, under which the dominant species will
change after a change of incoming light intensity
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

I_in_def = lambda lux: lambda lam: np.full(lam.shape, lux)
def k_spec():
    n_peak = 5
    l_p = np.random.uniform(400,700,n_peak)
    sigma_p = np.random.uniform(100,900, n_peak)
    # magnitude of peak, multiply by n_peaks, as not all peaks might be present
    gamma_p = np.random.uniform(0,1, n_peak)
    
    # absorption of pigments
    pigment = lambda lam: np.sum(gamma_p*np.exp(-(lam.reshape(-1,1)-l_p)**2\
                                                 /sigma_p),1)
    return pigment  
Ir = np.linspace(400,700,151)
pigment = k_spec()
plt.plot(Ir,pigment(Ir))
plt.show()
itera = 1000
fitness = 2e8*2**np.random.uniform(-1,1,(2,itera))/0.014
k = 2**np.random.uniform(-1,1,(2,itera))
pigment = lambda lam: k_spec()(lam).reshape(-1,1,1)*k

def equilibrium(fit,I_in = I_in_def(400/300), runs = 500):   
    # starting densities for iteration, shape = (npi, itera)
    equis = np.full(fit.shape, 1e10) # start of iteration
    
    lam, dx = np.linspace(400,700,51, retstep = True) #points for integration
    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = pigment(lam)
    I_in = I_in(lam)
    print(equis.shape, abs_points.shape)
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    for i in range(runs):
        if i ==runs-1:
            equis_old = equis.copy()
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = np.sum(equis*abs_points,1)
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = equis*abs_points #np.einsum('ni,li->nli', equis, abs_points)
        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = all_abs*(I_in[:,None]/tot_abs*(1-np.exp(-tot_abs)))[:,None]
        # fit*int(y_simps)
        equis = fit*simps(y_simps, dx = dx, axis = 0)
        # remove rare species
        equis[equis<1e4] = 0   
    hi = np.abs(equis-equis_old)/equis_old>1e-3
    return equis,hi

equi,hi = equilibrium(fitness)  
test = k*fitness
here = test[:,(equi[0]>equi[1])!=(test[0]>test[1])]
print(here)