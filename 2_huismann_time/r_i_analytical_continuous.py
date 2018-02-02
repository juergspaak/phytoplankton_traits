"""
@author: Jurg W. Spaak, jurg.spaak@unamur.be
Computes the boundary growth for both species with analytical solution
and continuous change in incoming light intensity
Assumes that I_out = 0 and background absorption is negligible
"""
import numpy as np
from scipy.integrate import simps
from differential_functions import cumsimps
import communities_analytical as com

def constant_I_r_i(species, I_in):
    # compute the boundary growth rate for constant incoming light intensity
    equi = com.equilibrium(species, I_in)
    k,l = species[[0,-1]]
    i,r = [0,1],[1,0]
    return l[i]*((k*equi)[i]/(k*equi)[r]-1)
       
def resident_density(species, I,period, acc = 1001):
    # check input
    
    if np.exp(period*np.amax(species[-1]))==np.inf:
        raise ValueError("Too long `period` in `resident_density`."+
        "The product l*period must be smaller than 700 to avoid overflow.")
    k,H,p,l = species # species parameters
    # time for numerical integration
    t,dt = np.linspace(0,period,acc,retstep = True)
    
    # equilibrium densities for incoming light
    W_r_star = com.equilibrium(species, I(t)[::2], "full")
    t.shape = -1,1,1
    int_fun = np.exp(l*t)/(H+I(t))*I.dt(t)
    # int_0^t e^(l*(s-t))/(H+I(s))*I.dt(s) ds, for t<period
    W_r_diff = p/(k*l)*cumsimps(int_fun,dx = dt, axis = 0, initial = 0)\
                    *np.exp(-l*t[::2])
    # W_r_diff has densitiy only at every second entry of t,due to integration
    dt *=2 
    # Finding W_r_diff(t=0), using W_r_diff(0)=W_r_diff(T)
    W_r_diff_0 = W_r_diff[-1]/(1-np.exp(-l*period))           
    # adding starting condition
    W_r_diff = W_r_diff + W_r_diff_0*np.exp(-l*t[::2])
    # computing real W_r_t
    W_r_t = W_r_star-W_r_diff

    return W_r_t, W_r_star,dt
    
def continuous_r_i(species, I,period, acc = 1001):
    # computes the boundary growth rates for invading species
    W_r_t, W_r_star,dt = resident_density(species, I, period,acc)
    i,r = [[0,1],[1,0]]
    k,l = species[[0,-1]]
    simple_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_star[:,r])-1,
                       dx = dt,axis = 0)*l[i]
    exact_r_i = simps(k[i]*W_r_star[:,i]/(k[r]*W_r_t[:,r])-1,
                       dx = dt,axis = 0)*l[i]
    return simple_r_i, exact_r_i
    
if __name__ == "__main__":
    # Plots that show correctness of programm
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    # defining I_in and plotting
    light_fluc = "sinus" # sinus for sinus shaped I_in, other-> toothsaw
    size,period = 40,10
    #size and period of fluctuation
    if light_fluc == "sinus":
        I = lambda t: size*np.sin(t/period*2*np.pi)+125 #sinus
        I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period
    else:
        peak_rel = 0.9 # relative location of peak of I_in
        T = peak_rel*period
        speed = peak_rel/(1-peak_rel)
        I = lambda t: size*((t%period<=T)*t%period/T+ #toothsaw
            (t%period>T)*(1-(t%period-T)*speed/T))+125-size/2
        I.dt = lambda t: size*((t%period<=T)/T-(t%period>T)/T*speed)
    
    # plots are: I_in, dI/dt, resident density, invader density
    fig,ax = plt.subplots(4,1,sharex = True,figsize = (9,9))
    time = np.linspace(0,3*period,1503)
    ax[-1].set_xlabel("time")
    
    # First plot; Incoming light intensity
    ax[0].plot(time, I(time))
    ax[0].set_ylabel("I(t)")
    
    # second plot; differentiate I_in over time
    ax[1].set_ylabel(r'$\frac{dI}{dt}$', fontsize = 16)
    ax[1].plot(time, I.dt(time))
    
    # computing resident densities
    species = com.gen_species(10) # random species
    # choose one of those random species
    i,j = np.random.randint(2),np.random.randint(10)
    # densities according to theory
    W_r_t, W_r_star, dt = resident_density(species,I,period,1001)
    
    def dWdt(W,t):
        k,H,p,l = species[:,i,j]
        W_star = p/(k*l)*np.log(1+I(t)/H)
        return (W_star-W)*l
    time2 = time[::50]
    # densities numerically checked
    sol_ode = odeint(dWdt,W_r_t[0,i,j],time2)
    
    # plot all these densities
    ax[2].plot(time,np.tile(W_r_star[:,i,j],3),label = "equilibrium")
    ax[2].plot(time,np.tile(W_r_t[:,i,j],3),label = "analytical")
    ax[2].plot(time2,sol_ode,'*',label = "numerical")
    ax[2].legend()
    ax[2].set_ylabel("resident density")

    # Invader densities
    ax[3].set_ylabel("invader growthrate")
    ax[3].plot(time, np.tile(W_r_star[:,1-i,j],3),label = "equilibrium")
    const = com.equilibrium(species, 125)
    ax[3].plot(time, np.full(1503,const[1-i,j]), label = "dens at const light")
    average = simps(W_r_t[:,1-i,j],time[:501])/period
    ax[3].plot(time, np.full(1503,average),label = "average over fluctuation")
    ax[3].legend(loc = "lower right")