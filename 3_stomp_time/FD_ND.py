"""
@author: J. W. Spaak, jurg.spaak@unamur.be
Compute the fitness and niche differences for the coexisting species"""


import numpy as np
from scipy.integrate import simps

from load_pigments import lambs, dlam
import multispecies_functions as mf
from generate_species import gen_com, n_diff_spe
from richness_computation import I_in_ref
import multispecies_functions as mf

def mono_equi(par, k_spec, I_in, runs = 1000):
    """Compute the monoculture equilibrium density of the species
    
    par: [phi,l]
        photosynthetic efficiency and loss rate of the species
    k_spec:
        absorption spectrum of the species
    I_in:
        Incoming light, can be an array of incoming lights
        
    Returns:
        equi: array with shape (len(I_ins), ) +phi.shape
            The equilibrium density of the species for each incoming light"""
    # number of different incoming light regimes
    if I_in.ndim == 1:
        I_in = I_in.reshape(1,-1)
        
    fitness = par[0]/par[1]
    I_in = I_in.view()
    I_in.shape = I_in.shape + (1,1)
    # Start of iteration, assume all light is absorbed
    equis = fitness*np.expand_dims(simps(I_in, dx = dlam,axis = -3),1)
    equis_fix = np.zeros(equis.shape)
    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = k_spec.copy()
    
    unfixed = np.full(fitness.shape[-1], True, dtype = bool)
    n = 20
    i = 0
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    while np.sum(unfixed)/equis.shape[-1]>0.01 and i<runs:
        # I_in*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = I_in*(1-np.exp(-equis*abs_points))
        # fit*int(y_simps)
        equis = fitness*np.expand_dims(simps(y_simps, dx = dlam, axis = -3),1)
        # remove rare species
        equis[equis<1] = 0
        if i % n==n-2:
            # to check stability of equilibrium in next run
            equis_old = equis.copy()
        if i % n==n-1:
            
            stable = np.logical_or(equis == 0, #no change or already 0
                                   np.abs((equis-equis_old)/equis)<1e-3)
            cond = np.logical_not(np.prod(stable, (0,1,2))) #at least one is unstable
            print(cond.shape, unfixed.shape, i)
            equis_fix[...,unfixed] = equis #copy the values
            # prepare for next runs
            unfixed[unfixed] = cond
            equis = equis[...,cond]
            abs_points = abs_points[...,cond]
            fitness = fitness[...,cond]
        i+=1
    #return only communities that found equilibrium
    return equis_fix
   
I_in = np.array([mf.I_in_def(40), I_in_ref(0),I_in_ref(2.5)])

[phi,l],k_spec,alpha = gen_com([1,5], 2, 10000,I_ins = I_in)  
  
par = np.array([phi,l])
runs = 1000

var_change = simps(k_spec, dx = dlam, axis = 0)

phi_n = par[0]*var_change
l_n = np.ones(phi_n.shape)*par[1]
k_spec_n = k_spec/var_change

equi_n = mono_equi([phi_n,l_n], k_spec_n, I_in, runs = 100)
equi = mono_equi(par, k_spec, I_in, runs = 100)

result = simps(k_spec_n[:,[1]]*(1-np.exp(-k_spec_n*equi_n[0,0,0]))/k_spec_n, 
               axis = 0, dx = dlam)
result = result/result[1]

I_in = I_in.reshape(-1,101,1,1)
f_0 = (phi_n*np.expand_dims(simps(I_in*k_spec_n,dx = dlam, 
                                 axis = 1),axis = 1)-l_n)[:,0]

tot_abs = (equi_n*k_spec_n)[:,:,::-1]
r_i = (phi_n*np.expand_dims(simps(I_in*k_spec_n/tot_abs*(1-np.exp(-tot_abs)),
                    axis = 1, dx = dlam),axis = 1)-l_n)[:,0]

tot_abs = (equi_n*k_spec_n)
equi_growth = (phi_n*np.expand_dims(simps(I_in*k_spec_n/tot_abs*(1-np.exp(-tot_abs)),
                    axis = 1, dx = dlam),axis = 1)-l_n)[:,0]

tot_abs = (equi_n[:,:,::-1]*k_spec_n)
f_N = (phi_n*np.expand_dims(simps(I_in*k_spec_n/tot_abs*(1-np.exp(-tot_abs)),
                    axis = 1, dx = dlam),axis = 1)-l_n)[:,0]

FD = -f_N/f_0
ND = (r_i-f_N)/(f_0-f_N)

fig, ax = plt.subplots(3,3, figsize = (11,11))

ax[0,0].set_title("Species 1")
ax[0,1].set_title("Species 2")
ax[1,0].set_ylabel("Light 2")

ND_all = np.linspace(np.amin(ND), np.amax(ND),100)
FD_all = ND_all/(1-ND_all)

r_i_rel = r_i/f_0

vmin, vmax = np.percentile(r_i_rel,[5,95])

for s in range(2):
    for l in range(len(I_in)):
        im = ax[l,s].scatter(ND[l,s], FD[l,s], linewidth = 0, c = r_i_rel[l,s],
            vmin = vmin, vmax = vmax)
        plt.colorbar(im,ax = ax[l,s])
        ax[l,s].plot(ND_all, FD_all)
        
        ax[l,-1].plot(lambs,I_in[l,:,0,0])
        ax[l,s].set_ylim([-1,np.percentile(FD[l,s],97.5)])

ax[1,1].set_ylim([-1,2])

ax[1,1].set_xlabel("Niche difference")
ax[0,0].set_ylabel("Fitness difference")
fig.savefig("ND and FD for spectrum model.png")
