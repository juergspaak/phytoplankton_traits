"""
@author: J.W.Spaak, jurg.spaak@unamur.be

Computes boundary growth rates for photoinhibition model"""
import numpy as np
import communities_numerical as com
from scipy.integrate import simps
from r_i_analytical_continuous import cumsimps, continuous_r_i

# incoming light is sinus shaped
period = 10
size = 450
I = lambda t: size*np.sin(t/period*2*np.pi)+550 #sinus
I.dt = lambda t:  size*np.cos(t/period*2*np.pi)*2*np.pi/period

def equilibrium(species, I_in):
    try:
        I_inv = I_in.view() 
        I_inv.shape = (-1,)+species[0].ndim*(1,)
    except AttributeError:
        I_inv = I_in
    
    k,p_max,I_k, I_opt, l = species
    
    # computation uses complex numbers
    discr = np.sqrt(1-4*I_k/I_opt+0j)
    I_1 = (-(1-2*I_k/I_opt)+discr)/(2*I_k/I_opt**2)
    I_2 = (-(1-2*I_k/I_opt)-discr)/(2*I_k/I_opt**2) 
    fun = lambda I: np.log(1-I_inv/I)
    prefactor = p_max/(l*k*(I_1-I_2))/I_k*I_opt**2
    return np.real(prefactor*(fun(I_1)-fun(I_2)))


def resident_density(species, I,period, acc = 1001):
    # check input
    if np.exp(period*np.amax(species[-1]))==np.inf:
        raise ValueError("Too long `period` in `resident_density`."+
        "The product l*period must be smaller than 700 to avoid overflow.")
    k,p,I_k, I_opt, l = species # species parameters
    # time for numerical integration
    t,dt = np.linspace(0,period,acc,retstep = True)
    
    # equilibrium densities for incoming light
    W_r_star = equilibrium(species, I(t)[::2])
    t.shape = -1,1,1
    a = I_k/I_opt**2
    b = 1-2*I_k/I_opt
    int_fun = np.exp(l*t)/(a*I(t)**2+b*I(t)+I_k)*I.dt(t)
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
    species,carbon, I_r = com.gen_species(com.photoinhibition_par,500)
    # choose one of those random species
    i,j = np.random.randint(2),np.random.randint(species.shape[-1])
    # densities according to theory
    W_r_t, W_r_star, dt = resident_density(species,I,period,1001)
    
    def dWdt(W,t):
        W_star = equilibrium(species[:,i,j], I(t))
        return (W_star-W)*species[-1,i,j]
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
    average = simps(W_r_t[:,1-i,j],time[:501])/period
    ax[3].plot(time, np.full(1503,average),label = "average over fluctuation")
    ax[3].legend(loc = "lower right")
