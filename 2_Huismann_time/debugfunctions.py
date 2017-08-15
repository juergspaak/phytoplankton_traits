"""
@author: J.W.Spaak

Debug functions
Contain functions to check, that the other functions work correctly
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_carbon_uptake(species, carbon, I_r, num = 5):
    plt.figure()
    carb_up = carbon(species, np.linspace(*I_r, 100), 'generating')
    for k in range(num):
        i = np.random.randint(2)
        j = np.random.randint(species.shape[-1])
        plt.plot(np.linspace(*I_r, 100),carb_up[i,j])
        
plot_carbon_uptake(spec, carb, [0,1000])
plot_carbon_uptake(spec2, carb2, [0,1000])