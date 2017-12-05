""" This program is to find a set of species, that allow two stable equilibria
with different species present at the equilibrium



doesn't work anymore, because alpha was redefined, change comp_beta_stable"""

import numpy as np
import multispecies_functions as equilibrium
import matplotlib.pyplot as plt

from load_pigments import real as l_pigs


def comp_beta_stable(alpha):
    beta = np.empty(alpha.shape[1:])
    mat = np.ones((alpha.shape[1], alpha.shape[1]))
    mat[0,-1] =0
    sol = np.zeros(alpha.shape[1])
    sol[0] = 1
    for i in range(alpha.shape[-1]):
        mat[1:] = alpha[...,i]
        try:
            beta[:,i] = np.linalg.solve(mat, sol)
        except np.linalg.linalg.LinAlgError:
            beta[:,i] = np.nan
    return beta
    
def plt_ncoex(equis):
    "see plot_pig_rich for documentation"
    spec_num = np.sum(equis>0,0)
    fig = plt.figure()
    plt.plot(np.linspace(0,100, len(spec_num)),sorted(spec_num))
    plt.ylabel("number coexisting species")
    plt.xlabel("percent")
    plt.show()
    return fig

# generate the communities
n_com = 5000 # number of communities
r_pig = 3 #richness of pigments in community
r_spec = r_pig+1 # richness of species, regional richness
r_pig_spec = 2 #richness of pigments in each species
fac = 2 #factor by which fitness can differ
phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com)) #photosynthetic efficiency
l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com)) # specific loss rate
fitness = phi/l # fitness
k_spec,alpha = equilibrium.spectrum_species(l_pigs, r_pig, 
                                            r_spec, n_com, r_pig_spec) #spectrum of the species
#equilibrium density
equis, unfixed = equilibrium.multispecies_equi(fitness, k_spec, runs = 1000)
# plot the number of coexisting species
plt_ncoex(equis[:,np.logical_not(unfixed)])
# find the cases where r_pig species coexist
surv = equis>0
full_surv = (np.sum(surv,0)==r_pig) & np.logical_not(unfixed)

#only care about the cases where all r_pig species coexist
k_spec = k_spec[...,full_surv]
alpha = alpha[...,full_surv]
fitness = fitness[...,full_surv]
surv = surv[...,full_surv]
# compute their linear combination, k[r_pig] = \sum_r beta[r]*k[r]
beta = comp_beta_stable(alpha)

#compute a matrix of betas:
betas = np.empty((r_spec, r_spec, np.sum(full_surv)))
r = slice(None)
betas[-1,r] = beta.copy()
betas[-1,-1] = 0

for j in range(r_spec-1):
    betas[j,r] = -betas[-1,r]/betas[-1,j] #k_i = \sum betas[j,r]*k_r, j!= r
    betas[j, -1] = 1/betas[-1,j]
    betas[j,j] = 0
# linear combination returning the spectrum of the species    
linear_combination = np.einsum('lsc,rsc->lrc',k_spec, betas)

# compute the invasion growthrates,fitness[i]*\sum_r betas[i,r]*1/fitness[r]-1
bound_growth = fitness*np.einsum('irc,rc->ic', betas, 1/fitness)-1

#species that can't invade
inv = bound_growth>0

# find the second equilibria
k_spec2 = np.empty(k_spec.shape-np.array([0,1,0]))
fitness2 = np.empty(fitness.shape -np.array([1,0]))
new_cases = inv|(~surv)
for i in range(fitness.shape[-1]):
    try:
        k_spec2[...,i] = k_spec[...,i][...,new_cases[...,i]]
        fitness2[...,i] = fitness[...,i][...,new_cases[...,i]]
    except ValueError:
        print(new_cases[...,i],i)
        k_spec2[...,i] = np.nan
        fitness2[...,i] = np.nan

equi2, unfixed = equilibrium.multispecies_equi(fitness2, k_spec2)
surv2 = equi2>0
print(np.amax(np.sum(surv2,0)))
plt_ncoex(equi2[:,np.logical_not(unfixed)])