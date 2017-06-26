"""shows that the approximation I_out = 0 is a good one if one of the species is
near its monocultural equilibrium
for this call show_I_out()

"""
import numpy as np
import generate_communities as com
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
    
def invader(species, I_T, t, start = 1):
    """computes the density of the invader species
    t is a vector, for which the density is computed
    start is the start density of the invader species"""
    W_now = com.equilibrium([I_T[1], I_T[1]], species, True)
    W_prev = com.equilibrium(I_T[0], species[1], True)
    l_i, l_r = species[:,-1]
    a = species[0][0]/species[1][0]*W_now[0]/W_now[1]
    b = l_i/l_r
    W_res = resident(species[1], I_T[1], t, W_prev)
    c_prime = start/W_prev**(a*b)
    return c_prime*(np.exp((a-1)*l_i*t)*W_res**(a*b))
    
def resident(species, I_now, t, start):
    """ computes the densitiy of the resident species, returns it to the times t
    species is one species, I_now is the current incoming light, t is a vector of times
    start is the start densitiy of the resident species (W_r(0)=start)"""
    t = np.array(t)
    [k,H,p_max,l] = species
    W_star = p_max/(l*k)*np.log(1+I_now/H) #equilibrium density
    return W_star -(W_star-start)*np.exp(-l*t) #exponentially approaches the equilibrial value W_star 


def invader_growth(biomass,t,species, start_res, I_in, res_fun,approx = 1):
    """computes the growth rate for the invading species at time t
    x is the species density, species = [invader, resident], I_in is the current incoming light"""
    [k,H,p_max,l] = species[0]
    #W_res = resident(species[1], I_in, t, start_res)
    W_res = res_fun(t)
    return (p_max*np.log((H+I_in)/(H+approx*I_in*np.exp(-species[1][-1]*W_res)))
            /(species[1][0]*W_res)-l)*biomass
    
def resident_growth(biomass,t,species, I_in):
    k,H,p_max,l = species[1]
    dwdt = p_max/k*np.log((H+I_in)/(H+I_in*np.e**(-k*biomass)))-l*biomass
    return dwdt
    

def show_I_out():        
    species = com.find_species()
    start_inv = 1
    t_end = 50
    time = np.linspace(0,t_end,300)
    I_T = np.random.uniform(50,200,2)
    start_res = com.equilibrium(I_T[0], species[1])
    
    approx_inv = np.array(invader(species, I_T, time, start_inv))
    approx_res = np.array(resident(species[1], I_T[1], time,start_res))
    
    
    exact_res = odeint(resident_growth,start_res,time, (species, I_T[1]))
    exact_res = [e[0] for e in exact_res]
    res_fun_prov = interp1d(time, exact_res,'cubic')
    res_fun = lambda t: res_fun_prov(min(t,50))
    exact_inv = np.array(odeint(invader_growth,start_inv,time, (species, start_res, I_T[1],res_fun, 1)))
    res_fun = lambda t: resident(species[1], I_T[1], t,start_res)
    
    fig1 = plt.figure(1)
    plt.plot(time[0::15],approx_res[0::15], '^')
    plt.plot(time,exact_res)
    fig1.gca().set_ylabel("density of resident", fontsize=14)
    fig1.gca().set_xlabel("time", fontsize=14)
    plt.legend(["approximated", "exact"], loc = "best")
    
    fig2 = plt.figure(2)
    plt.plot(time,[1-exact_res[i]/approx_res[i] for i in range(len(time))],'r')
    fig2.gca().set_ylabel("relative error, resident", fontsize=14)
    fig2.gca().set_xlabel("time", fontsize=14)
    
    fig3 = plt.figure(3)
    plt.plot(time[0::15],approx_inv[0::15], '^')
    plt.plot(time,exact_inv)
    fig3.gca().set_ylabel("density of invader", fontsize=14)
    fig3.gca().set_xlabel("time", fontsize=14)
    plt.legend(["approximated", "exact"], loc = "best")
    
    fig4 = plt.figure(4)
    plt.plot(time,[1-exact_inv[i]/approx_inv[i] for i in range(len(time))],'r')
    fig4.gca().set_ylabel("relative error, invader", fontsize=14)
    fig4.gca().set_xlabel("time", fontsize=14)
    
    
    """ #saving:
    save_fig("I_out, zero, resident", fig1, typ = "pdf")
    save_fig("I_out, zero, invader", fig3, typ = "pdf")
    save_fig("I_out, zero, invader, rel error", fig4, typ = "pdf")"""