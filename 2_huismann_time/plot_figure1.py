"""
@author: Jurg W. Spaak
plots figure 1
"""

import matplotlib.pyplot as plt
import numpy as np

import analytical_communities as com
import analytical_r_i_continuous as ana

species = com.gen_species(10000)
plt.figure(figsize = (9,9))
"""
for T in 2*np.pi*np.array([50,50**0.5,1]):
    size  = 40
    I = lambda t: size*np.sin(t/T*2*np.pi)+125 #sinus
    I.dt = lambda t:  size*np.cos(t/T*2*np.pi)*2*np.pi/T
    simple_r_i, exact_r_i = ana.continuous_r_i(species, I, T)
    plt.plot(*exact_r_i,',', label="per:"+str(T)+", size:" +str(size))
    print(np.sum(np.sum(exact_r_i>0, axis = 0)==2))
#plt.axis([-1,1,-1,1])
plt.grid()
plt.legend()"""

plt.figure(figsize = (9,9))

for size in [1,40**0.5,40]:
    t = 2*np.pi*10
    I = lambda t: size*np.sin(t/T*2*np.pi)+125 #sinus
    I.dt = lambda t:  size*np.cos(t/T*2*np.pi)*2*np.pi/T
    simple_r_i, exact_r_i = ana.continuous_r_i(species, I, T)
    plt.plot(*exact_r_i,',', label="per:"+str(T)+", size:" +str(size))
    print(np.sum(np.sum(exact_r_i>0, axis = 0)==2))
#plt.axis([-1,1,-1,1])
plt.grid()
plt.legend()
"""
plt.figure(figsize = (9,9))

for T in 2*np.pi*np.array([0.5,1,2,20]):
    for size in [5,10,40]:
        speed = 10
        period = T*(1/speed+1)
        t,dt = np.linspace(0,5*period,1001, retstep = True)
        I = lambda t: size*((t%period<=T)*t%period/T+ #toothsaw
                    (t%period>T)*(1-(t%period-T)*speed/T))+125-size/2
        I.dt = lambda t: size*((t%period<=T)/T-(t%period>T)/T*speed)
        simple_r_i, exact_r_i = ana.continuous_r_i(species, I, T)
        plt.plot(*exact_r_i,',', label="per:"+str(period)+", size:" +str(size))
plt.axis([-1,1,-1,1])
plt.legend()

print(np.sum(np.sum(exact_r_i>0, axis = 0)==2))"""