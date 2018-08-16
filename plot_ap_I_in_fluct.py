"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the appendix,
for more information check the description in the appendix

Plot different cases of fluctiating incoming light cases
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import I_in_functions as I_inf
from pigments import lambs

I_in = [I_inf.I_in_def(30,450), I_inf.I_in_def(50,650)]

period = 10 # period length
# creates fucntion I_in(t)
I_int = I_inf.fluc_continuous(p_loc = period)

fig, ax = plt.subplots(3,2, figsize = (9,11), sharex = True, sharey = True)
times = np.linspace(0,5,6)
colors = cm.rainbow(np.linspace(0, 1, len(times)))

ax[0,0].set_ylabel(r"light intensity [$\mu\, mol\, m^{-2} s^{-1}$]", 
    fontsize = 14)
ax[1,0].set_ylabel(r"light intensity [$\mu\, mol\, m^{-2} s^{-1}$]", 
    fontsize = 14)
ax[2,0].set_ylabel(r"light intensity [$\mu\, mol\, m^{-2} s^{-1}$]", 
    fontsize = 14)


ax[2,0].set_xlabel("Wavelength [nm]")
ax[2,1].set_xlabel("Wavelength [nm]")

# The first constant light
ax[0,0].set_title("A; Constant light 1")
ax[0,0].plot(lambs, I_in[0])

# Second constant light
ax[0,1].set_title("B; Constant light 2")
ax[0,1].plot(lambs, I_in[1])

# Sinus shaped switching between the constant cases
ax[1,0].set_title("C Sinus shaped switching")
for i,t in enumerate(times):
    if i<3:
        ax[1,0].plot(lambs, I_inf.fluc_nconst(I_in, period, "sinus")(t),
            color = colors[i], label = "t = {}".format(int(t)))
    else:
        ax[1,0].plot(lambs, I_inf.fluc_nconst(I_in, period, "sinus")(t),
            color = colors[i])
    
# Sawtooth case
ax[1,1].set_title("D; Sawtooth switching")
for i,t in enumerate(times):
    if i>=3:
        ax[1,1].plot(lambs, I_inf.fluc_nconst(I_in, period, "linear")(t),
                color = colors[i], label = "t = {}".format(int(t)))
    else:
        ax[1,1].plot(lambs, I_inf.fluc_nconst(I_in, period, "linear")(t),
                color = colors[i])

ax[2,0].set_title("E; Rectangular switching")
for i,t in enumerate(times):
    ax[2,0].plot(lambs, I_inf.fluc_nconst(I_in, period, "Rect")(t),
            color = colors[i])
    
ax[2,1].set_title("F; Continuous change")
for i,t in enumerate(times):
    plt.plot(lambs, I_int(t),
            color = colors[len(colors)-i-1])

ax[1,0].legend(loc = "upper left")
ax[1,1].legend(loc = "upper left")

fig.savefig("Figure,ap_I_in_fluct.pdf")