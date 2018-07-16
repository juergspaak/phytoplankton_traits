"""
@author: J. W. Spaak, jurg.spaak@unamur.be

create different fluctuating incoming light functions
for illustrations of the different fluctuating types see: plot_ap_I_In_fluct.py
"""

import numpy as np
from scipy.integrate import simps

from pigments import lambs, dlam

def I_in_def(lux, loc = 550, sigma = 50):
    """ returns a gaussian kernel incoming light function"""
    if sigma == 0: # constant incoming light intensity
        return np.full(lambs.shape, lux, dtype = "float")
    else:
        prelim = np.exp(-(lambs-loc)**2/(2*sigma**2))
        return lux*prelim/simps(prelim, dx = dlam)

def I_in_composite(n_peaks):
    "Return light that is a composite of n_peaks gaussian kernels"
    luxs = np.random.uniform(30,50,n_peaks)
    sigmas = 2**np.random.uniform(4,9,n_peaks) # ragnes from 16-512
    locs = np.random.uniform(400,700,n_peaks)
    I_in = np.average([I_in_def(luxs[i], locs[i], sigmas[i]) 
                    for i in range(n_peaks)], axis = 0)
    return I_in
 
def sinus(t):
    #sinus wave fluctuations
    res = (1-np.cos(2*np.pi*t))/2
    return np.array([res,1-res])
     
###############################################################################
# funtctions that return a callable I_in(t)

def fluc_continuous(w_loc = [450, 650], w_lux = [30, 50],
                p_loc = 10, p_lux = None, sigma = 50):
    """returns I_in(t), that continously fluctuates over the spectrum
    
    w_loc: Minimal and maximal value for `loc`in mf.I_in_def
    w_lux: Minimal and maximal value for `lux`in mf.I_in_def
    p_loc: period length of fluctuation in loc
    p_lux: period length of fluctuation in lux
    sigma: passed on to mf.I_in_def"""
    if p_lux is None: p_lux = p_loc
    w_loc = np.asanyarray(w_loc)
    w_lux = np.asanyarray(w_lux)
    def I_in(t):
        """incoming light at time t"""
        loc_t = (sinus((t%p_loc)/p_loc)*w_loc).sum()
        lux_t = (sinus((t%p_lux)/p_lux)*w_lux).sum()
        return I_in_def(lux_t, loc_t, sigma)
    return I_in

# default value for fluc_nconst
def_I_ins = np.array([I_in_def(40/300, 450,50), 
                      I_in_def(40/300, 650,50)])
    
def fluc_nconst(I_ins = def_I_ins, period = 10, fluc_case = "sinus"):
    """returns I_in(t), that averages over the I_ins
    
    I_in(t) is a weighted average over the I_ins
    
    I_ins: array of shape (n,101), different incoming light
    period: period length
    fluc_case: ("sinus", "linear" of None). How the light flucutates"""
    if fluc_case == "sinus":
        fluc_fun = lambda t: (1-np.cos(t*np.pi))/2
    elif fluc_case == "linear":
        fluc_fun = lambda t: t
    else: # step function
        fluc_fun = lambda t: np.round(t)
    id_I_in, dI = np.linspace(0,period,num = len(I_ins)+1, retstep = True)
    id_I_in.shape = -1,1
    def I_in(t):
        """incoming light at time t"""
        t = t%period
        # proportion of each incoming light
        prop_I = (t>=id_I_in-dI)*(t<id_I_in)*fluc_fun(t%dI/dI)+\
                (t>=id_I_in)*(t<id_I_in+dI)*(1-fluc_fun(t%dI/dI))
        prop_I[0] += prop_I[-1]
        return (prop_I[:-1]*I_ins).sum(axis = 0)
    return I_in
    
if __name__ == "__main__":# examples
    # For further examples see plot_ap_I_in_fluct.py
    
    import matplotlib.pyplot as plt
    period = 10 # period length
    # creates fucntion I_in(t)
    I_in_cont = fluc_continuous()
    time = np.linspace(0,period, period+1)
    for t in time:
        plt.plot(lambs,I_in_cont(t))
    plt.title("Continuous change of spectrum")
    plt.show()
    
    # case where we switch between 2 I_in functions, linear
    plt.figure()
    I_in_step = fluc_nconst(fluc_case = "linear")
    time = np.linspace(0,period, period+1)
    for t in time:
        plt.plot(lambs,I_in_step(t))
    plt.title("linear function fluctuations")
    plt.show()
    
    # example with composite multiple I_ins continuous
    time = np.linspace(0,period, period +2)
    fig, ax = plt.subplots(2,2,sharex = True,figsize = (9,9))
    ax[1,0].set_xlabel("nm")
    ax[1,1].set_xlabel("nm")
    I_ins = np.array([I_in_composite(2), I_in_composite(3), 
                      I_in_composite(10)])
    I_in_step = fluc_nconst(I_ins)
    
    # plot the different lights
    ax[0,0].plot(lambs, I_ins[0])
    ax[0,0].set_title("2 peaks")
    ax[0,1].plot(lambs, I_ins[1])
    ax[0,1].set_title("3 peaks")
    ax[1,0].plot(lambs, I_ins[2])
    ax[1,0].set_title("10 peaks")
    
    for t in time:
        ax[1,1].plot(lambs, I_in_step(t))
        ax[1,1].set_title("Wheighted average over time")
    