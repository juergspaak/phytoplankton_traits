"plots which parameters are importatn for coexistnece"

import numpy as np
import matplotlib.pyplot as plt

n_spec = 100000 # numer of species to compute
species = com.gen_species(n_spec)

invasion_fluct = np.empty(n_spec)
for i in range(10):
    ind = np.arange(i*10000, (i+1)*10000)
    invasion_fluct[ind] = np.amin(ana.continuous_r_i(species[...,ind], I,period)[1], axis = 0)\
                         /period #normalize by period
    print(i)

n_spec = species.shape[-1]
R = species[2]/species[3] # pmax/l
K = species[1]

pos = invasion_fluct>0
op = np.argmax(R,axis = 0)
gl = np.argmin(R,axis = 0)

R_ratio = R[gl,np.arange(n_spec)]/R[op,np.arange(n_spec)]

plt.figure()
plt.plot(np.linspace(0,100,len(R_ratio)),sorted(R_ratio))
plt.plot(np.linspace(0,100,sum(pos)),sorted(R_ratio[pos]),'o')
plt.title("ratio of R")

plt.figure()
plt.plot(K[op, np.arange(n_spec)], K[gl,np.arange(n_spec)],',')
plt.plot(K[op[pos],np.arange(n_spec)[pos]],
           K[gl[pos],np.arange(n_spec)[pos]],'ro')
plt.xlabel("oportunist K")
plt.ylabel("gleaner K")

plt.figure()
plt.loglog(R[op, np.arange(n_spec)], R[gl,np.arange(n_spec)],',')
plt.loglog(R[op[pos],np.arange(n_spec)[pos]],
           R[gl[pos],np.arange(n_spec)[pos]],'ro')
plt.xlabel("oportunist R")
plt.ylabel("gleaner R")

plt.figure()
plt.title("distribution of K,gleaner")
plt.plot(np.linspace(0,100,len(R_ratio)),sorted(K[op, np.arange(n_spec)]))
plt.plot(np.linspace(0,100,sum(pos)),sorted(K[op[pos],np.arange(n_spec)[pos]]),'o')

tit = ["k", "h", "p", "l"]
opgl = ["op", "gl"]
for i,par in enumerate(species):
    plt.figure()
    plt.plot(par[op, np.arange(n_spec)], par[gl,np.arange(n_spec)],',')
    plt.plot(par[op[pos],np.arange(n_spec)[pos]],
           par[gl[pos],np.arange(n_spec)[pos]],'ro')
    plt.xlabel("op")
    plt.ylabel("gl")
    plt.title(tit[i])
    
    
    for j,tata in enumerate([op,gl]):
        plt.figure()
        plt.title(tit[i]+opgl[j])
        plt.plot(np.linspace(0,100,n_spec),sorted(par[tata, np.arange(n_spec)]))
        plt.plot(np.linspace(0,100,sum(pos)),sorted(par[tata[pos],np.arange(n_spec)[pos]]),'o')

plt.figure()
plt.plot(species[2,gl,np.arange(n_spec)], species[3, op,np.arange(n_spec)],',')
plt.plot(species[2,gl[pos],np.arange(n_spec)[pos]], species[3, op[pos],np.arange(n_spec)[pos]],'ro')
plt.show()