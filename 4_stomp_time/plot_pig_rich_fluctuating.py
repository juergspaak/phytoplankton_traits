"""@author: J.W.Spaak
plots richness of species depending on pigment richness"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from fluctuating_spectra import fluctuating_richness

import sys
sys.path.append("../3_different_pigments")
import load_pigments as lp
from load_pigments import lambs, dlam
import multispecies_functions as mf


richnesses = np.empty((10,4,10))
fig, ax = plt.subplots()
color = ['b','c', 'r','g']
setoff = 0.5*np.array([-1,-0.5,0.5,1])
regress_x = [[],[],[],[]]
regress_y = [[],[],[],[]]
for i,r_pig in list(enumerate(range(2,22,2))):
    richnesses[i] = fluctuating_richness(r_pig,n_com = 100, 
                r_pig_spec = min(3, r_pig))
    print(r_pig)
    for case in range(4):
        ax.scatter(np.full(10,r_pig+setoff[case]), range(1,11),
                   s = 20*richnesses[i,case], c = color[case])
        for j in range(10):
            regress_x[case] += int(1000*richnesses[i,case,j])*[r_pig]
            regress_y[case] += int(1000*richnesses[i,case,j])*[j+1]
ax.axis([1,21,0.9, np.argmax((richnesses==0).sum(axis = (0,1)))+0.1])
x = np.arange(2,21)
for case in range(4):
    slope, intercept, c,d,e = linregress(regress_x[case], regress_y[case])
    plt.plot(x,intercept+slope*x, color[case])

    