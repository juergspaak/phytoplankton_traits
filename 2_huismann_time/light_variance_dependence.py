"""
@author: J.W. Spaak

Find the dependence of the boundary growth rate in dependence of the variance
in incoming light"""

import numpy as np
import analytical_r_i_continuous as r_i_con
import analytical_communities as com

import matplotlib.pyplot as plt
from timeit import default_timer as timer

species = com.gen_species(num = int(1e3))
T = 10
period = 20
speed = T/(period-T)

                             
t = np.linspace(0,5*period,1001)                             

sizes = [0,10,20,30,40,50,60,70,80]
r_is = []
for size in sizes:
    I = lambda t: size*((t%period<=T)*t%period/T+
                    (t%period>T)*(1-(t%period-T)*speed/T))+130-size/2
    I.dt = lambda t: size*((t%period<=T)/T-(t%period>T)/T*speed)

    r_is.append(r_i_con.continuous_r_i(species, I,t)[1])
r_is = np.asanyarray(r_is)

bal = com.find_balance(species)
balanced = np.logical_and(129.5<bal,bal<130.5)
