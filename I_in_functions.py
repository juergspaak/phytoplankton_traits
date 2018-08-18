"""
@author: J. W. Spaak, jurg.spaak@unamur.be

create different fluctuating incoming light functions
for illustrations of the different fluctuating types see: plot_ap_I_In_fluct.py
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simps

from pigments import lambs, dlam


###############################################################################
# Sun light data

sun_data = pd.read_csv("sunlight.csv")
sun_spectrum = {}    
for key in ["usual", "direct full", "blue sky", "clouded"]:
    x = sun_data["lambs, " + key]
    y = sun_data[key]
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    sun_spectrum[key] = interp1d(x,y)(lambs)
    sun_spectrum[key] = sun_spectrum[key]/simps(sun_spectrum[key],dx = dlam)

def sun_light(lux = [20,200], period = 10, sky = "blue sky"):
    """returns whithe light that changes total intensity over time
    
    lux: array of shape (2), minimal and maximal lux of the incoming light
    period: float, length of period
    
    Returns: I_in(t), function returning incoming light at given time t"""
    

    def I_in(t):
        "gives white light at time t"
        
        alpha = (np.cos(2*t*np.pi/period)+1)/2
        lux_t = alpha*lux[0] + (1-alpha)*lux[1]
        return lux_t*sun_spectrum[sky]
        
    return I_in
    
###############################################################################
# background absorption

background = pd.read_csv("background_absorption.csv")

k_water = background["lambs, water"], background["water"]
ind = np.isfinite(k_water[0])
k_water = interp1d(k_water[0][ind], k_water[1][ind])(lambs)

k_gli_tri = interp1d(background["lambs, GliTri"], background["GliTri"])(lambs)

# different background absorptions depending on the environment
k_BG_rel = {"ocean": k_water,
        "baltic sea": k_water + k_gli_tri,
        "peat lake": k_water + k_gli_tri/k_gli_tri[50]*2}

# mixing depth in meters depending on the environment
zm = {"ocean": 60, "baltic sea": 10, "peat lake": 1}

# relative background absorption not needed, only total background absorption
k_BG = {key: k_BG_rel[key]*zm[key] for key in zm.keys()}


if False and __name__ == "__main__":# examples
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
    