# -*- coding: utf-8 -*-
"""
shows, that the mp approximation is a good one
"""
import matplotlib.pyplot as plt
import numpy as np
import generate_communities as com
import short_and_long_period as per

def P_mp(P_ran = [-2,1], acc = 15):
    """ shows, that the mp approximation is a good one"""
    mps = []
    res = com.find_species()[0]
    W_av = com.equilibrium(125, res)
    m_r, q_r = per.linear_envi(res)
    Ps = 10**np.linspace(*P_ran, acc)
    mrs = []
    mrs2 = []
    for i in Ps:
        m,q = dist_resident(res,W_av, i)
        mps.append(m)
        mrs.append(m_r*((1-np.exp(-res[-1]*i))/(1+np.exp(-res[-1]*i)))) #asymptotic behavior
        mrs2.append(m_r*((1-np.exp(-res[-1]*i))/(1+np.exp(-res[-1]*i)))**0.5) #good approximation for "large" P
    plt.loglog(Ps, mps)
    plt.loglog(Ps, mrs, '^')
    plt.loglog(Ps, mrs2, ':')
    plt.grid()
    plt.legend(["exact", "asymptotic", "approx for large P"], loc = "best")
    return np.array([mps,mrs, mrs2])
  
def dist_resident(resident, W_av, P = 5, I_in = [50,200], itera = 10000,
                          plot_bool = False):
    """gives a distribution of W_p(T)"""
    l =resident[-1]
    W_r = np.zeros(itera)
    lights = np.random.uniform(I_in[0], I_in[1], itera)
    lights[0] = np.average(I_in)
    W_r[0] = com.equilibrium(lights[0], resident)
    for i in range(itera-1):
        W_r_star = com.equilibrium(lights[i+1], resident, True)
        
        W_r[i+1] = W_r_star-(W_r_star-W_r[i])*np.exp(-l*P)
    adap_light = sorted(lights)-np.average(I_in)
    W_adap =W_r -W_av 
    cov = sum(adap_light*sorted(W_adap))
    var = sum(adap_light**2)
    m = cov/var # linear regression with 0 intercept, enforcing that m*Iav +q = Wav
    q = W_av-m*np.average(I_in)
    return m,q
    

mps, mrs, mrs2 = P_mp()
    